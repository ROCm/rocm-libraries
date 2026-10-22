// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

 // Tensor.BF8.d2 0, (base=&0, sizes={&8 &16 }, strides={&24 &32 })
 // T_LOAD_TILED 1 Source 0
 // Tensor.FP8.d2 2, (base=&40, sizes={&48 &56 }, strides={&64 &72 })
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
 // Tensor_0_pointer: PointerGlobal: BF8(offset: 0, size: 8, read_write)
 // Tensor_0_size_0: Value: Int64(offset: 8, size: 8, read_only)
 // Tensor_0_size_1: Value: Int64(offset: 16, size: 8, read_only)
 // Tensor_0_stride_0: Value: Int64(offset: 24, size: 8, read_only)
 // Tensor_0_stride_1: Value: Int64(offset: 32, size: 8, read_only)
 // Tensor_2_pointer: PointerGlobal: FP8(offset: 40, size: 8, read_write)
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
 // KernelArg{Tensor_0_pointer, PointerGlobal: BF8, read_write, CommandArgument(Tensor_0_pointer)PG(c 0), o:8, s:8, preloaded}
 // KernelArg{Tensor_0_size_0_1, Value: Int64, CommandArgument(Tensor_0_size_0)I64(c 0), o:16, s:8, preloaded}
 // KernelArg{Tensor_2_stride_1_3, Value: Int64, CommandArgument(Tensor_2_stride_1)I64(c 0), o:24, s:8, preloaded}
 // KernelArg{Tensor_2_pointer, PointerGlobal: FP8, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
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
.globl RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2,@function
// Allocated Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1
// Allocated Preloaded argument block: 14 SGPRs (Value: Raw32): s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15
// Allocated Workgroup Index X: 1 SGPR (Value: UInt32): s16
// Allocated Workgroup Index Y: 1 SGPR (Value: UInt32): s17
// Allocated Packed Workitem Index: 1 VGPR (Value: Raw32): v0
// Allocated Workitem Index X: 1 VGPR (Value: UInt32): v1
// Allocated Workitem Index Y: 1 VGPR (Value: UInt32): v2
RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2:
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
RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_exec_begin:
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
 // KernelArg{Tensor_2_pointer, PointerGlobal: FP8, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
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
 // Tensor_0_pointer: SGPR PointerGlobal: BF8 x 1: s[4:5]
 // Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7]
 // Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]
 // Tensor_2_pointer: SGPR PointerGlobal: FP8 x 1: s[10:11]
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
 // RemoveImplicitScheduling
 // OrderMultiplyNodes
 // Simplify
 // ScheduleMultiplyAndLDS
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
 // (op 1) generate({827, 879, 932, 985, 1058, 1097, 1121, 1145})
 // (op 1) NOP(827) BEGIN
 // (op 827) generate({})
 // (op 827) end: generate({})
 // (op 1) NOP(827) END
 // (op 1) NOP(879) BEGIN
 // (op 879) generate({})
 // (op 879) end: generate({})
 // (op 1) NOP(879) END
 // (op 1) NOP(932) BEGIN
 // (op 932) generate({})
 // (op 932) end: generate({})
 // (op 1) NOP(932) END
 // (op 1) NOP(985) BEGIN
 // (op 985) generate({})
 // (op 985) end: generate({})
 // (op 1) NOP(985) END
 // (op 1) NOP(1058) BEGIN
 // (op 1058) generate({})
 // (op 1058) end: generate({})
 // (op 1) NOP(1058) END
 // (op 1) NOP(1097) BEGIN
 // (op 1097) generate({})
 // (op 1097) end: generate({})
 // (op 1) NOP(1097) END
 // (op 1) NOP(1121) BEGIN
 // (op 1121) generate({})
 // (op 1121) end: generate({})
 // (op 1) NOP(1121) END
 // (op 1) NOP(1145) BEGIN
 // (op 1145) generate({})
 // (op 1145) end: generate({})
 // (op 1) NOP(1145) END
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(831) BEGIN
 // (op 831) Assign dim(588) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 588: v**UNALLOCATED**
 // (op 831) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag588: VGPR Value: UInt32 x 1: (unallocated)
 // (op 831) reg expression
 // (op 831) BEGIN: Flatten[1]
 // (op 831) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(883) BEGIN
 // (op 883) Assign dim(604) = {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 604: v**UNALLOCATED**
 // (op 883) Generate {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag604: VGPR Value: UInt32 x 1: (unallocated)
 // (op 883) reg expression
 // (op 883) BEGIN: Flatten[1]
 // (op 883) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}(936) BEGIN
 // (op 936) Assign dim(620) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 620: v**UNALLOCATED**
 // (op 936) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag620: VGPR Value: UInt32 x 1: (unallocated)
 // (op 936) reg expression
 // (op 936) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}(989) BEGIN
 // (op 989) Assign dim(636) = {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 636: v**UNALLOCATED**
 // (op 989) Generate {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag636: VGPR Value: UInt32 x 1: (unallocated)
 // (op 989) reg expression
 // (op 989) BEGIN: Flatten[1]Tile
 // (op 989) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(1062) BEGIN
 // (op 1062) Assign dim(658) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 658: v**UNALLOCATED**
 // (op 1062) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag658: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1062) reg expression
 // (op 1062) BEGIN: Flatten[1]
 // (op 1062) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(1101) BEGIN
 // (op 1101) Assign dim(670) = {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 670: v**UNALLOCATED**
 // (op 1101) Generate {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag670: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1101) reg expression
 // (op 1101) BEGIN: Flatten[1]
 // (op 1101) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}(1125) BEGIN
 // (op 1125) Assign dim(678) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 678: v**UNALLOCATED**
 // (op 1125) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag678: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1125) reg expression
 // (op 1125) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}(1149) BEGIN
 // (op 1149) Assign dim(686) = {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 686: v**UNALLOCATED**
 // (op 1149) Generate {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag686: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1149) reg expression
 // (op 1149) BEGIN: Flatten[1]Tile
 // (op 1149) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 831): v0
v_and_b32 v0, 63, v1 // (op 831) 
 // (op 831) END: Flatten[1]
 // (op 831) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 831): v3
v_lshrrev_b32 v3, 4, v0 // (op 831) 
 // (op 831) BEGIN: Flatten[1]
 // (op 831) {Flatten[1]: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 831): v4
v_and_b32 v4, 1, v3 // (op 831) 
 // (op 831) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 831): v3

 // (op 831) BEGIN: Flatten[1]
 // (op 831) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 831): v3
v_and_b32 v3, 15, v1 // (op 831) 
 // (op 831) END: Flatten[1]
 // (op 831) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 831): v5
v_lshl_add_u32 v5, v4, 4, v3 // (op 831) 
 // (op 831) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 831): v3

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 831): v4

 // (op 831) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 831): v3
v_lshrrev_b32 v3, 6, v1 // (op 831) 
 // (op 831) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 831): v4
v_lshl_add_u32 v4, v3, 5, v5 // (op 831) 
 // (op 831) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 831): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 831): v3

 // (op 831) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 831): v3
v_lshrrev_b32 v3, 5, v0 // (op 831) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 831): v0

 // (op 831) BEGIN: TileTile
 // (op 831) {TileTile: ShiftL(v3:U32, 4:U32)U32}
 // (op 831) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 831)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 831)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 831): v0
v_lshlrev_b32 v0, 4, v3 // (op 831) 
 // (op 831) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 831): v3

 // (op 831) BEGIN: Tile
// Allocated DataFlowTag588: 1 VGPR (Value: UInt32) (op 831): v3
v_lshl_add_u32 v3, v4, 6, v0 // (op 831) 
 // (op 831) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 831): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 831): v4

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(831) END
// Allocated : 1 VGPR (Value: UInt32) (op 883): v0
v_and_b32 v0, 63, v1 // (op 883) 
 // (op 883) END: Flatten[1]
 // (op 883) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 883): v4
v_lshrrev_b32 v4, 4, v0 // (op 883) 
 // (op 883) BEGIN: Flatten[1]
 // (op 883) {Flatten[1]: BitwiseAnd(v4:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 883): v5
v_and_b32 v5, 1, v4 // (op 883) 
 // (op 883) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 883): v4

 // (op 883) BEGIN: Flatten[1]
 // (op 883) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 883): v4
v_and_b32 v4, 15, v1 // (op 883) 
 // (op 883) END: Flatten[1]
 // (op 883) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 883): v6
v_lshl_add_u32 v6, v5, 4, v4 // (op 883) 
 // (op 883) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 883): v4

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 883): v5

 // (op 883) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 883): v4
v_lshrrev_b32 v4, 5, v0 // (op 883) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 883): v0

 // (op 883) BEGIN: TileTile
 // (op 883) {TileTile: ShiftL(v4:U32, 4:U32)U32}
 // (op 883) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 883)         value (VGPR Value: UInt32 x 1: v4) 
 // (op 883)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 883): v0
v_lshlrev_b32 v0, 4, v4 // (op 883) 
 // (op 883) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 883): v4

 // (op 883) BEGIN: Tile
// Allocated DataFlowTag604: 1 VGPR (Value: UInt32) (op 883): v4
v_lshl_add_u32 v4, v6, 6, v0 // (op 883) 
 // (op 883) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 883): v0

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 883): v6

 // (op 1) Assign VGPR {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(883) END
// Allocated : 1 VGPR (Value: UInt32) (op 936): v0
v_lshrrev_b32 v0, 6, v1 // (op 936) 
 // (op 936) BEGIN: Flatten[1]
 // (op 936) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 936): v5
v_and_b32 v5, 31, v1 // (op 936) 
 // (op 936) END: Flatten[1]
 // (op 936) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 936): v6
v_lshl_add_u32 v6, v0, 5, v5 // (op 936) 
 // (op 936) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 936): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 936): v0

 // (op 936) BEGIN: Flatten[1]
 // (op 936) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 936): v0
v_and_b32 v0, 63, v1 // (op 936) 
 // (op 936) END: Flatten[1]
 // (op 936) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 936): v5
v_lshrrev_b32 v5, 5, v0 // (op 936) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 936): v0

 // (op 936) BEGIN: Tile
// Allocated DataFlowTag620: 1 VGPR (Value: UInt32) (op 936): v0
v_lshl_add_u32 v0, v6, 1, v5 // (op 936) 
 // (op 936) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 936): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 936): v6

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}(936) END
// Allocated : 1 VGPR (Value: UInt32) (op 989): v5
v_and_b32 v5, 31, v1 // (op 989) 
 // (op 989) END: Flatten[1]Tile
 // (op 989) BEGIN: Flatten[1]
 // (op 989) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 989): v6
v_and_b32 v6, 63, v1 // (op 989) 
 // (op 989) END: Flatten[1]
 // (op 989) LogicalShiftR({Flatten[1]: v6:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 989): v7
v_lshrrev_b32 v7, 5, v6 // (op 989) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 989): v6

 // (op 989) BEGIN: Tile
// Allocated DataFlowTag636: 1 VGPR (Value: UInt32) (op 989): v6
v_lshl_add_u32 v6, v5, 1, v7 // (op 989) 
 // (op 989) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 989): v7

 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 989): v5

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}(989) END
// Allocated : 1 VGPR (Value: UInt32) (op 1062): v5
v_and_b32 v5, 63, v1 // (op 1062) 
 // (op 1062) END: Flatten[1]
 // (op 1062) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1062): v7
v_lshrrev_b32 v7, 4, v5 // (op 1062) 
 // (op 1062) BEGIN: Flatten[1]
 // (op 1062) {Flatten[1]: BitwiseAnd(v7:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1062): v8
v_and_b32 v8, 1, v7 // (op 1062) 
 // (op 1062) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1062): v7

 // (op 1062) BEGIN: Flatten[1]
 // (op 1062) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1062): v7
v_and_b32 v7, 15, v1 // (op 1062) 
 // (op 1062) END: Flatten[1]
 // (op 1062) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 1062): v9
v_lshl_add_u32 v9, v8, 4, v7 // (op 1062) 
 // (op 1062) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1062): v7

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1062): v8

 // (op 1062) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1062): v7
v_lshrrev_b32 v7, 6, v1 // (op 1062) 
 // (op 1062) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 1062): v8
v_lshl_add_u32 v8, v7, 5, v9 // (op 1062) 
 // (op 1062) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1062): v9

 // Freeing : 1 VGPR (Value: UInt32) (op 1062): v7

 // (op 1062) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1062): v7
v_lshrrev_b32 v7, 5, v5 // (op 1062) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1062): v5

 // (op 1062) BEGIN: TileTile
 // (op 1062) {TileTile: ShiftL(v7:U32, 4:U32)U32}
 // (op 1062) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1062)         value (VGPR Value: UInt32 x 1: v7) 
 // (op 1062)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1062): v5
v_lshlrev_b32 v5, 4, v7 // (op 1062) 
 // (op 1062) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 1062): v7

 // (op 1062) BEGIN: Tile
// Allocated DataFlowTag658: 1 VGPR (Value: UInt32) (op 1062): v7
v_lshl_add_u32 v7, v8, 6, v5 // (op 1062) 
 // (op 1062) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1062): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1062): v8

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(104)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(104)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(1062) END
// Allocated : 1 VGPR (Value: UInt32) (op 1101): v5
v_and_b32 v5, 63, v1 // (op 1101) 
 // (op 1101) END: Flatten[1]
 // (op 1101) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
 // (op 1) Deallocate{}(1391) BEGIN
 // (op 1391) Deallocate 104
 // Deleting tag 104
 // (op 1) Deallocate{}(1391) END
// Allocated : 1 VGPR (Value: UInt32) (op 1101): v8
v_lshrrev_b32 v8, 4, v5 // (op 1101) 
 // (op 1101) BEGIN: Flatten[1]
 // (op 1101) {Flatten[1]: BitwiseAnd(v8:U32, 1:U32)U32}
 // (op 1) Assign VGPR 64:U32(832) BEGIN
 // (op 832) Assign dim(589) = 64:U32
 // (op 832) Assign stride expression
 // (op 1) Assign VGPR 64:U32(832) END
 // (op 1) Assign VGPR 64:U32(1063) BEGIN
 // (op 1063) Assign dim(659) = 64:U32
 // (op 1063) Assign stride expression
 // (op 1) Assign VGPR 64:U32(1063) END
 // (op 1) NOP(828) BEGIN
 // (op 828) generate({})
 // (op 828) end: generate({})
 // (op 1) NOP(828) END
 // (op 1) NOP(1059) BEGIN
 // (op 1059) generate({})
 // (op 1059) end: generate({})
 // (op 1) NOP(1059) END
 // (op 1) Assign VGPR 1:U32(836) BEGIN
 // (op 836) Assign dim(591) = 1:U32
 // (op 836) Assign stride expression
 // (op 1) Assign VGPR 1:U32(836) END
 // (op 1) Assign VGPR 1:U32(1067) BEGIN
 // (op 1067) Assign dim(661) = 1:U32
 // (op 1067) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1067) END
// Allocated : 1 VGPR (Value: UInt32) (op 1101): v9
v_and_b32 v9, 1, v8 // (op 1101) 
 // (op 1101) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1101): v8

 // (op 1101) BEGIN: Flatten[1]
 // (op 1101) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1101): v8
v_and_b32 v8, 15, v1 // (op 1101) 
 // (op 1101) END: Flatten[1]
 // (op 1101) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 1101): v10
v_lshl_add_u32 v10, v9, 4, v8 // (op 1101) 
 // (op 1101) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1101): v8

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1101): v9

 // (op 1101) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1101): v8
v_lshrrev_b32 v8, 5, v5 // (op 1101) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1101): v5

 // (op 1101) BEGIN: TileTile
 // (op 1101) {TileTile: ShiftL(v8:U32, 4:U32)U32}
 // (op 1101) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1101)         value (VGPR Value: UInt32 x 1: v8) 
 // (op 1101)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1101): v5
v_lshlrev_b32 v5, 4, v8 // (op 1101) 
 // (op 1101) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 1101): v8

 // (op 1101) BEGIN: Tile
// Allocated DataFlowTag670: 1 VGPR (Value: UInt32) (op 1101): v8
v_lshl_add_u32 v8, v10, 6, v5 // (op 1101) 
 // (op 1101) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1101): v5

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1101): v10

 // (op 1) Assign VGPR {Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 15:U32)U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(170)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(1101) END
// Allocated : 1 VGPR (Value: UInt32) (op 1125): v5
v_lshrrev_b32 v5, 6, v1 // (op 1125) 
 // (op 1125) BEGIN: Flatten[1]
 // (op 1125) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 1) Deallocate{}(1394) BEGIN
 // (op 1394) Deallocate 170
 // Deleting tag 170
 // (op 1) Deallocate{}(1394) END
// Allocated : 1 VGPR (Value: UInt32) (op 1125): v9
v_and_b32 v9, 31, v1 // (op 1125) 
 // (op 1125) END: Flatten[1]
 // (op 1125) BEGIN: Tile
 // (op 1) Assign VGPR 64:U32(884) BEGIN
 // (op 884) Assign dim(605) = 64:U32
 // (op 884) Assign stride expression
 // (op 1) Assign VGPR 64:U32(884) END
 // (op 1) Assign VGPR 64:U32(1102) BEGIN
 // (op 1102) Assign dim(671) = 64:U32
 // (op 1102) Assign stride expression
 // (op 1) Assign VGPR 64:U32(1102) END
 // (op 1) NOP(880) BEGIN
 // (op 880) generate({})
 // (op 880) end: generate({})
 // (op 1) NOP(880) END
 // (op 1) NOP(1098) BEGIN
 // (op 1098) generate({})
 // (op 1098) end: generate({})
 // (op 1) NOP(1098) END
 // (op 1) Assign VGPR 1:U32(888) BEGIN
 // (op 888) Assign dim(607) = 1:U32
 // (op 888) Assign stride expression
 // (op 1) Assign VGPR 1:U32(888) END
 // (op 1) Assign VGPR 1:U32(1106) BEGIN
 // (op 1106) Assign dim(673) = 1:U32
 // (op 1106) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1106) END
// Allocated : 1 VGPR (Value: UInt32) (op 1125): v10
v_lshl_add_u32 v10, v5, 5, v9 // (op 1125) 
 // (op 1125) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1125): v9

 // Freeing : 1 VGPR (Value: UInt32) (op 1125): v5

 // (op 1125) BEGIN: Flatten[1]
 // (op 1125) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1125): v5
v_and_b32 v5, 63, v1 // (op 1125) 
 // (op 1125) END: Flatten[1]
 // (op 1125) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1125): v9
v_lshrrev_b32 v9, 5, v5 // (op 1125) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1125): v5

 // (op 1125) BEGIN: Tile
// Allocated DataFlowTag678: 1 VGPR (Value: UInt32) (op 1125): v5
v_lshl_add_u32 v5, v10, 1, v9 // (op 1125) 
 // (op 1125) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 1125): v9

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1125): v10

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(236)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(236)U32, 63:U32)U32}, 5:U32)U32)U32}(1125) END
// Allocated : 1 VGPR (Value: UInt32) (op 1149): v9
v_and_b32 v9, 31, v1 // (op 1149) 
 // (op 1149) END: Flatten[1]Tile
 // (op 1149) BEGIN: Flatten[1]
 // (op 1149) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Deallocate{}(1397) BEGIN
 // (op 1397) Deallocate 236
 // Deleting tag 236
 // (op 1) Deallocate{}(1397) END
// Allocated : 1 VGPR (Value: UInt32) (op 1149): v10
v_and_b32 v10, 63, v1 // (op 1149) 
 // (op 1149) END: Flatten[1]
 // (op 1149) LogicalShiftR({Flatten[1]: v10:U32}, 5:U32)U32
 // (op 1) Assign VGPR 2:U32(937) BEGIN
 // (op 937) Assign dim(621) = 2:U32
 // (op 937) Assign stride expression
 // (op 1) Assign VGPR 2:U32(937) END
 // (op 1) Assign VGPR 2:U32(1126) BEGIN
 // (op 1126) Assign dim(679) = 2:U32
 // (op 1126) Assign stride expression
 // (op 1) Assign VGPR 2:U32(1126) END
 // (op 1) NOP(933) BEGIN
 // (op 933) generate({})
 // (op 933) end: generate({})
 // (op 1) NOP(933) END
 // (op 1) NOP(1122) BEGIN
 // (op 1122) generate({})
 // (op 1122) end: generate({})
 // (op 1) NOP(1122) END
 // (op 1) Assign VGPR 1:U32(941) BEGIN
 // (op 941) Assign dim(623) = 1:U32
 // (op 941) Assign stride expression
 // (op 1) Assign VGPR 1:U32(941) END
 // (op 1) Assign VGPR 1:U32(1130) BEGIN
 // (op 1130) Assign dim(681) = 1:U32
 // (op 1130) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1130) END
// Allocated : 1 VGPR (Value: UInt32) (op 1149): v11
v_lshrrev_b32 v11, 5, v10 // (op 1149) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1149): v10

 // (op 1149) BEGIN: Tile
// Allocated DataFlowTag686: 1 VGPR (Value: UInt32) (op 1149): v10
v_lshl_add_u32 v10, v9, 1, v11 // (op 1149) 
 // (op 1149) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 1149): v11

 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 1149): v9

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(296)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(296)U32, 63:U32)U32}, 5:U32)U32)U32}(1149) END
 // (op 1) Deallocate{}(1400) BEGIN
 // (op 1400) Deallocate 296
 // Deleting tag 296
 // (op 1) Deallocate{}(1400) END
 // (op 1) Assign VGPR 2:U32(990) BEGIN
 // (op 990) Assign dim(637) = 2:U32
 // (op 990) Assign stride expression
 // (op 1) Assign VGPR 2:U32(990) END
 // (op 1) Assign VGPR 2:U32(1150) BEGIN
 // (op 1150) Assign dim(687) = 2:U32
 // (op 1150) Assign stride expression
 // (op 1) Assign VGPR 2:U32(1150) END
 // (op 1) NOP(986) BEGIN
 // (op 986) generate({})
 // (op 986) end: generate({})
 // (op 1) NOP(986) END
 // (op 1) NOP(1146) BEGIN
 // (op 1146) generate({})
 // (op 1146) end: generate({})
 // (op 1) NOP(1146) END
 // (op 1) Assign VGPR 1:U32(994) BEGIN
 // (op 994) Assign dim(639) = 1:U32
 // (op 994) Assign stride expression
 // (op 1) Assign VGPR 1:U32(994) END
 // (op 1) Assign VGPR 1:U32(1154) BEGIN
 // (op 1154) Assign dim(689) = 1:U32
 // (op 1154) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1154) END
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
 // (op 1) Deallocate{user_Float_Value_14}(1443) BEGIN
 // (op 1443) Deallocate user_Float_Value_14
 // Freeing user_Float_Value_14: 1 SGPR (Value: Raw32): s14

 // (op 1) Deallocate{user_Float_Value_14}(1443) END
 // (op 1) Deallocate{user_Float_Value_12}(1445) BEGIN
 // (op 1445) Deallocate user_Float_Value_12
 // Freeing user_Float_Value_12: 1 SGPR (Value: Raw32): s15

 // (op 1) Deallocate{user_Float_Value_12}(1445) END
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
 // (op 344) Scope(534) BEGIN
 // (op 534) Lock Scope 534
 // Lock Branch
 // (op 534) generate({529})
 // (op 534) NOP(529) BEGIN
 // (op 529) generate({})
 // (op 529) end: generate({})
 // (op 534) NOP(529) END
 // (op 534) Assign VGPR 64:U64(539) BEGIN
 // (op 539) Assign dim(502) = 64:U64
 // (op 539) Assign stride expression
 // (op 534) Assign VGPR 64:U64(539) END
 // (op 534) NOP(530) BEGIN
 // (op 530) generate({})
 // (op 530) end: generate({})
 // (op 534) NOP(530) END
 // (op 534) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(542) BEGIN
 // (op 542) Assign dim(503) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}
 // tag 503: v**UNALLOCATED**
 // (op 542) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64} into DataFlowTag503: VGPR Value: UInt64 x 1: (unallocated)
 // (op 542) reg expression
 // (op 542) reg expression
 // (op 542) reg expression
 // (op 542) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 542): v12
v_lshrrev_b32 v12, 2, v1 // (op 542) 
 // (op 542) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 542): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 542) 
 // (op 542) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 542): v12

 // (op 542) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 542): v12
v_and_b32 v12, -4, v1 // (op 542) 
 // (op 542) BEGIN: Flatten[1]
 // (op 542) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 542): v14
v_sub_u32 v14, v1, v12 // (op 542) 
 // (op 542) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 542): v12

 // (op 542) BEGIN: TileTile
 // (op 542) {TileTile: ShiftL({Flatten[1]: v14:U32}, 4:U32)U32}
 // (op 542) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 542)         value (Flatten[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 542)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 542): v12
v_lshlrev_b32 v12, 4, v14 // (op 542) 
 // (op 542) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 542): v14

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 542): v14, v15
v_mov_b32 v15, 0 // (op 542) convert
v_mov_b32 v14, v12 // (op 542) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 542): v12

 // (op 542) BEGIN: Split
 // (op 542) Multiply: dest (DataFlowTag503: VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 542)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 542)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated DataFlowTag503: 2 VGPRs (Value: UInt64) (op 542): v16, v17
v_mul_lo_u32 v17, v13, s3 // (op 542) most significant: low of low * high
 // (op 542) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 542): v12
v_mul_hi_u32 v12, v13, s2 // (op 542) most significant: high of low * low
v_mul_lo_u32 v16, v13, s2 // (op 542) least significant: low of low * low
v_add_u32 v17, v17, v12 // (op 542) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 542): v12

 // (op 542) Add: dest (DataFlowTag503: VGPR Value: UInt64 x 1: v[16:17]) = 
 // (op 542)      lhs (DataFlowTag503: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 542)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 542): s0, s1
v_add_co_u32 v16, s[0:1], v16, v14 // (op 542) least significant half
v_addc_co_u32 v17, s[0:1], v17, v15, s[0:1] // (op 542) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 542): s0, s1

 // (op 542) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 542): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 542): v13

 // (op 534) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(542) END
 // (op 534) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(543) BEGIN
 // (op 543) Assign dim(504) = Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64
 // (op 543) Assign stride expression
 // (op 534) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(543) END
 // (op 534) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(544) BEGIN
 // (op 544) Assign dim(497) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 497: s**UNALLOCATED**
 // (op 544) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag497: SGPR Buffer: None x 1: (unallocated)
 // (op 544) reg expression
 // (op 544) reg expression
 // (op 544) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 544) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 544)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 544)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 544): s0, s1
s_mul_i32 s1, s2, s7 // (op 544) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 544): s14
s_mul_i32 s14, s3, s6 // (op 544) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 544) least significant
// Allocated : 1 SGPR (Value: Int32) (op 544): s15
s_mul_hi_u32 s15, s2, s6 // (op 544) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 544) most significant: sum
s_add_u32 s1, s1, s15 // (op 544) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 544): s15

 // Freeing : 1 SGPR (Value: Int32) (op 544): s14

 // (op 544) reg expression
 // (op 544) reg expression
// Allocated DataFlowTag497: 4 SGPRs (Buffer: None) (op 544): s40, s41, s42, s43
s_mov_b32 s40, s4 // (op 544) reg expression
s_mov_b32 s41, s5 // (op 544) reg expression
s_mov_b32 s42, s0 // (op 544) reg expression
s_mov_b32 s43, 131072 // (op 544) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 544): s0, s1

 // (op 534) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(544) END
 // (op 534) NOP(531) BEGIN
 // (op 531) generate({})
 // (op 531) end: generate({})
 // (op 534) NOP(531) END
 // (op 534) Assign VGPR 1:U64(549) BEGIN
 // (op 549) Assign dim(506) = 1:U64
 // (op 549) Assign stride expression
 // (op 534) Assign VGPR 1:U64(549) END
 // (op 534) NOP(552) BEGIN
 // (op 552) generate({})
 // (op 552) end: generate({})
 // (op 534) NOP(552) END
 // (op 534) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(561) BEGIN
 // (op 561) Assign dim(507) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}
 // tag 507: v**UNALLOCATED**
 // (op 561) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64} into DataFlowTag507: VGPR Value: UInt64 x 1: (unallocated)
 // (op 561) reg expression
 // (op 561) reg expression
 // (op 561) reg expression
 // (op 561) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 561): v12
v_lshrrev_b32 v12, 2, v1 // (op 561) 
 // (op 561) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 561): v13
v_lshl_add_u32 v13, s16, 5, v12 // (op 561) 
 // (op 561) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 561): v12

 // (op 561) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 561): v12
v_and_b32 v12, -4, v1 // (op 561) 
 // (op 561) BEGIN: Flatten[1]
 // (op 561) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 561): v14
v_sub_u32 v14, v1, v12 // (op 561) 
 // (op 561) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 561): v12

 // (op 561) BEGIN: TileTile
 // (op 561) {TileTile: ShiftL({Flatten[1]: v14:U32}, 4:U32)U32}
 // (op 561) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 561)         value (Flatten[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 561)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 561): v12
v_lshlrev_b32 v12, 4, v14 // (op 561) 
 // (op 561) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 561): v14

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 561): v14, v15
v_mov_b32 v15, 0 // (op 561) convert
v_mov_b32 v14, v12 // (op 561) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 561): v12

 // (op 561) BEGIN: Split
 // (op 561) Multiply: dest (DataFlowTag507: VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 561)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 561)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated DataFlowTag507: 2 VGPRs (Value: UInt64) (op 561): v18, v19
v_mul_lo_u32 v19, v13, s3 // (op 561) most significant: low of low * high
 // (op 561) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 561): v12
v_mul_hi_u32 v12, v13, s2 // (op 561) most significant: high of low * low
v_mul_lo_u32 v18, v13, s2 // (op 561) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 561) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 561): v12

 // (op 561) Add: dest (DataFlowTag507: VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 561)      lhs (DataFlowTag507: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 561)      rhs (TileTile: VGPR Value: UInt64 x 1: v[14:15])
// Allocated : 2 SGPRs (Value: Bool64) (op 561): s0, s1
v_add_co_u32 v18, s[0:1], v18, v14 // (op 561) least significant half
v_addc_co_u32 v19, s[0:1], v19, v15, s[0:1] // (op 561) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 561): s0, s1

 // (op 561) END: Split
 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 561): v14, v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 561): v13

 // (op 534) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U64)U64}(561) END
 // (op 534) Assign VGPR 64:U64(562) BEGIN
 // (op 562) Assign dim(508) = 64:U64
 // (op 562) Assign stride expression
 // (op 534) Assign VGPR 64:U64(562) END
 // (op 534) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(563) BEGIN
 // (op 563) Assign dim(497) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 563) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // (op 563) reg expression
 // (op 563) reg expression
 // (op 563) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 563) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 563)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 563)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 563): s0, s1
s_mul_i32 s1, s2, s7 // (op 563) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 563): s14
s_mul_i32 s14, s3, s6 // (op 563) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 563) least significant
// Allocated : 1 SGPR (Value: Int32) (op 563): s15
s_mul_hi_u32 s15, s2, s6 // (op 563) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 563) most significant: sum
s_add_u32 s1, s1, s15 // (op 563) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 563): s15

 // Freeing : 1 SGPR (Value: Int32) (op 563): s14

 // (op 563) reg expression
 // (op 563) reg expression
s_mov_b32 s40, s4 // (op 563) reg expression
s_mov_b32 s41, s5 // (op 563) reg expression
s_mov_b32 s42, s0 // (op 563) reg expression
s_mov_b32 s43, 131072 // (op 563) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 563): s0, s1

 // (op 534) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(563) END
 // (op 534) NOP(554) BEGIN
 // (op 554) generate({})
 // (op 554) end: generate({})
 // (op 534) NOP(554) END
 // (op 534) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(568) BEGIN
 // (op 568) Assign dim(510) = Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64
 // (op 568) Assign stride expression
 // (op 534) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(568) END
 // (op 534) NOP(555) BEGIN
 // (op 555) generate({})
 // (op 555) end: generate({})
 // (op 534) NOP(555) END
 // (op 534) Assign VGPR 1:U64(571) BEGIN
 // (op 571) Assign dim(512) = 1:U64
 // (op 571) Assign stride expression
 // (op 534) Assign VGPR 1:U64(571) END
 // (op 534) NOP(596) BEGIN
 // (op 596) generate({})
 // (op 596) end: generate({})
 // (op 534) NOP(596) END
 // (op 534) Assign VGPR 64:U64(603) BEGIN
 // (op 603) Assign dim(521) = 64:U64
 // (op 603) Assign stride expression
 // (op 534) Assign VGPR 64:U64(603) END
 // (op 534) NOP(597) BEGIN
 // (op 597) generate({})
 // (op 597) end: generate({})
 // (op 534) NOP(597) END
 // (op 534) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(606) BEGIN
 // (op 606) Assign dim(522) = {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}
 // tag 522: v**UNALLOCATED**
 // (op 606) Generate {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64} into DataFlowTag522: VGPR Value: UInt64 x 1: (unallocated)
 // (op 606) reg expression
 // (op 606) reg expression
 // (op 606) reg expression
 // (op 606) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 606): v12
v_and_b32 v12, -4, v1 // (op 606) 
 // (op 606) BEGIN: Flatten[1]
 // (op 606) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 606): v13
v_sub_u32 v13, v1, v12 // (op 606) 
 // (op 606) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 606): v12

 // (op 606) BEGIN: TileTile
 // (op 606) {TileTile: ShiftL({Flatten[1]: v13:U32}, 4:U32)U32}
 // (op 606) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 606)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 606)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 606): v12
v_lshlrev_b32 v12, 4, v13 // (op 606) 
 // (op 606) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 606): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 606): v14, v15
v_mov_b32 v15, 0 // (op 606) convert
v_mov_b32 v14, v12 // (op 606) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 606): v12

 // (op 606) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 606): v12
v_lshrrev_b32 v12, 2, v1 // (op 606) 
 // (op 606) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 606): v13
v_lshl_add_u32 v13, s17, 5, v12 // (op 606) 
 // (op 606) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 606): v12

 // (op 606) Multiply({Tile: v13:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 606) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 606)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 606)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 606): v20, v21
v_mul_lo_u32 v21, v13, s9 // (op 606) most significant: low of low * high
 // (op 606) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 606): v12
v_mul_hi_u32 v12, v13, s8 // (op 606) most significant: high of low * low
v_mul_lo_u32 v20, v13, s8 // (op 606) least significant: low of low * low
v_add_u32 v21, v21, v12 // (op 606) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 606): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 606): v13

 // (op 606) BEGIN: Split
 // (op 606) {Split: Add({TileTile: v[14:15]:U64}, v[20:21]:I64)U64}
 // (op 606) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 606)      lhs (TileTile: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 606)      rhs (VGPR Value: Int64 x 1: v[20:21])
// Allocated DataFlowTag522: 2 VGPRs (Value: UInt64) (op 606): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 606): s0, s1
v_add_co_u32 v12, s[0:1], v14, v20 // (op 606) least significant half
v_addc_co_u32 v13, s[0:1], v15, v21, s[0:1] // (op 606) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 606): s0, s1

 // (op 606) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 606): v20, v21

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 606): v14, v15

 // (op 534) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(606) END
 // (op 534) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(607) BEGIN
 // (op 607) Assign dim(523) = Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64
 // (op 607) Assign stride expression
 // (op 534) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(607) END
 // (op 534) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(608) BEGIN
 // (op 608) Assign dim(516) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 516: s**UNALLOCATED**
 // (op 608) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag516: SGPR Buffer: None x 1: (unallocated)
 // (op 608) reg expression
 // (op 608) reg expression
 // (op 608) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 608) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 608)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 608)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 608): s0, s1
s_mul_i32 s1, s8, s13 // (op 608) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 608): s14
s_mul_i32 s14, s9, s12 // (op 608) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 608) least significant
// Allocated : 1 SGPR (Value: Int32) (op 608): s15
s_mul_hi_u32 s15, s8, s12 // (op 608) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 608) most significant: sum
s_add_u32 s1, s1, s15 // (op 608) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 608): s15

 // Freeing : 1 SGPR (Value: Int32) (op 608): s14

 // (op 608) reg expression
 // (op 608) reg expression
// Allocated DataFlowTag516: 4 SGPRs (Buffer: None) (op 608): s44, s45, s46, s47
s_mov_b32 s44, s10 // (op 608) reg expression
s_mov_b32 s45, s11 // (op 608) reg expression
s_mov_b32 s46, s0 // (op 608) reg expression
s_mov_b32 s47, 131072 // (op 608) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 608): s0, s1

 // (op 534) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(608) END
 // (op 534) NOP(598) BEGIN
 // (op 598) generate({})
 // (op 598) end: generate({})
 // (op 534) NOP(598) END
 // (op 534) Assign VGPR 1:U64(613) BEGIN
 // (op 613) Assign dim(525) = 1:U64
 // (op 613) Assign stride expression
 // (op 534) Assign VGPR 1:U64(613) END
 // (op 534) NOP(616) BEGIN
 // (op 616) generate({})
 // (op 616) end: generate({})
 // (op 534) NOP(616) END
 // (op 534) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(625) BEGIN
 // (op 625) Assign dim(526) = {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}
 // tag 526: v**UNALLOCATED**
 // (op 625) Generate {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64} into DataFlowTag526: VGPR Value: UInt64 x 1: (unallocated)
 // (op 625) reg expression
 // (op 625) reg expression
 // (op 625) reg expression
 // (op 625) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 625): v14
v_and_b32 v14, -4, v1 // (op 625) 
 // (op 625) BEGIN: Flatten[1]
 // (op 625) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v14:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 625): v15
v_sub_u32 v15, v1, v14 // (op 625) 
 // (op 625) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 625): v14

 // (op 625) BEGIN: TileTile
 // (op 625) {TileTile: ShiftL({Flatten[1]: v15:U32}, 4:U32)U32}
 // (op 625) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 625)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 625)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 625): v14
v_lshlrev_b32 v14, 4, v15 // (op 625) 
 // (op 625) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 625): v15

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 625): v20, v21
v_mov_b32 v21, 0 // (op 625) convert
v_mov_b32 v20, v14 // (op 625) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 625): v14

 // (op 625) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 625): v14
v_lshrrev_b32 v14, 2, v1 // (op 625) 
 // (op 625) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 625): v15
v_lshl_add_u32 v15, s17, 5, v14 // (op 625) 
 // (op 625) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 625): v14

 // (op 625) Multiply({Tile: v15:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 625) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 625)           lhs (Tile: VGPR Value: UInt32 x 1: v15) 
 // (op 625)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 625): v22, v23
v_mul_lo_u32 v23, v15, s9 // (op 625) most significant: low of low * high
 // (op 625) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 625): v14
v_mul_hi_u32 v14, v15, s8 // (op 625) most significant: high of low * low
v_mul_lo_u32 v22, v15, s8 // (op 625) least significant: low of low * low
v_add_u32 v23, v23, v14 // (op 625) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 625): v14

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 625): v15

 // (op 625) BEGIN: Split
 // (op 625) {Split: Add({TileTile: v[20:21]:U64}, v[22:23]:I64)U64}
 // (op 625) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 625)      lhs (TileTile: VGPR Value: UInt64 x 1: v[20:21]) 
 // (op 625)      rhs (VGPR Value: Int64 x 1: v[22:23])
// Allocated DataFlowTag526: 2 VGPRs (Value: UInt64) (op 625): v14, v15
// Allocated : 2 SGPRs (Value: Bool64) (op 625): s0, s1
v_add_co_u32 v14, s[0:1], v20, v22 // (op 625) least significant half
v_addc_co_u32 v15, s[0:1], v21, v23, s[0:1] // (op 625) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 625): s0, s1

 // (op 625) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 625): v22, v23

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 625): v20, v21

 // (op 534) Assign VGPR {Split: Add(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -4:I)U32)U32}, 4:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(625) END
 // (op 534) Assign VGPR 64:U64(626) BEGIN
 // (op 626) Assign dim(527) = 64:U64
 // (op 626) Assign stride expression
 // (op 534) Assign VGPR 64:U64(626) END
 // (op 534) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(627) BEGIN
 // (op 627) Assign dim(516) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 627) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // (op 627) reg expression
 // (op 627) reg expression
 // (op 627) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 627) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 627)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 627)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 627): s0, s1
s_mul_i32 s1, s8, s13 // (op 627) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 627): s14
s_mul_i32 s14, s9, s12 // (op 627) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 627) least significant
// Allocated : 1 SGPR (Value: Int32) (op 627): s15
s_mul_hi_u32 s15, s8, s12 // (op 627) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 627) most significant: sum
s_add_u32 s1, s1, s15 // (op 627) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 627): s15

 // Freeing : 1 SGPR (Value: Int32) (op 627): s14

 // (op 627) reg expression
 // (op 627) reg expression
s_mov_b32 s44, s10 // (op 627) reg expression
s_mov_b32 s45, s11 // (op 627) reg expression
s_mov_b32 s46, s0 // (op 627) reg expression
s_mov_b32 s47, 131072 // (op 627) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 627): s0, s1

 // (op 534) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(627) END
 // (op 534) NOP(618) BEGIN
 // (op 618) generate({})
 // (op 618) end: generate({})
 // (op 534) NOP(618) END
 // (op 534) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(632) BEGIN
 // (op 632) Assign dim(529) = Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64
 // (op 632) Assign stride expression
 // (op 534) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(632) END
 // (op 534) NOP(619) BEGIN
 // (op 619) generate({})
 // (op 619) end: generate({})
 // (op 534) NOP(619) END
 // (op 534) Assign VGPR 1:U64(635) BEGIN
 // (op 635) Assign dim(531) = 1:U64
 // (op 635) Assign stride expression
 // (op 534) Assign VGPR 1:U64(635) END
 // (op 534) NOP(660) BEGIN
 // (op 660) generate({})
 // (op 660) end: generate({})
 // (op 534) NOP(660) END
 // (op 534) Assign VGPR 2:U64(667) BEGIN
 // (op 667) Assign dim(540) = 2:U64
 // (op 667) Assign stride expression
 // (op 534) Assign VGPR 2:U64(667) END
 // (op 534) NOP(661) BEGIN
 // (op 661) generate({})
 // (op 661) end: generate({})
 // (op 534) NOP(661) END
 // (op 534) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}(670) BEGIN
 // (op 670) Assign dim(541) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}
 // tag 541: v**UNALLOCATED**
 // (op 670) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32} into DataFlowTag541: VGPR Value: UInt32 x 1: (unallocated)
 // (op 670) reg expression
 // (op 670) reg expression
 // (op 670) reg expression
 // (op 670) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 670) convert
// Allocated : 1 VGPR (Value: UInt32) (op 670): v20
v_lshrrev_b32 v20, 1, v1 // (op 670) 
 // (op 670) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 670): v21
v_lshl_add_u32 v21, s16, 5, v20 // (op 670) 
 // (op 670) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 670): v20

 // (op 670) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 670): v20
v_and_b32 v20, -2, v1 // (op 670) 
 // (op 670) BEGIN: Flatten[1]TileTile
 // (op 670) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v20:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 670): v22
v_sub_u32 v22, v1, v20 // (op 670) 
 // (op 670) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 670): v20

 // (op 670) BEGIN: Split
 // Freeing Manually loaded argument block for range 56-128: 18 SGPRs (Value: Raw32): s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37

// Allocated DataFlowTag541: 1 VGPR (Value: UInt32) (op 670): v20
s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v20, v21, s20 // (op 670) 
v_add_u32 v20, v20, v22 // (op 670) 
 // (op 670) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 670): v22

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 670): v21

 // (op 534) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}(670) END
 // (op 534) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(671) BEGIN
 // (op 671) Assign dim(542) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 671) Assign stride expression
 // (op 534) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(671) END
 // (op 534) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(672) BEGIN
 // (op 672) Assign dim(535) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 535: s**UNALLOCATED**
 // (op 672) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag535: SGPR Buffer: None x 1: (unallocated)
 // (op 672) reg expression
 // (op 672) reg expression
 // (op 672) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 672) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 672)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 672)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 672): s0, s1
s_mul_i32 s1, s20, s7 // (op 672) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 672): s14
s_mul_i32 s14, s21, s6 // (op 672) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 672) least significant
// Allocated : 1 SGPR (Value: Int32) (op 672): s15
s_mul_hi_u32 s15, s20, s6 // (op 672) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 672) most significant: sum
s_add_u32 s1, s1, s15 // (op 672) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 672): s15

 // Freeing : 1 SGPR (Value: Int32) (op 672): s14

 // (op 672) reg expression
 // (op 672) reg expression
// Allocated DataFlowTag535: 4 SGPRs (Buffer: None) (op 672): s48, s49, s50, s51
s_mov_b32 s48, s22 // (op 672) reg expression
s_mov_b32 s49, s23 // (op 672) reg expression
s_mov_b32 s50, s0 // (op 672) reg expression
s_mov_b32 s51, 131072 // (op 672) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 672): s0, s1

 // (op 534) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(672) END
 // (op 534) NOP(662) BEGIN
 // (op 662) generate({})
 // (op 662) end: generate({})
 // (op 534) NOP(662) END
 // (op 534) Assign VGPR 1:U64(677) BEGIN
 // (op 677) Assign dim(544) = 1:U64
 // (op 677) Assign stride expression
 // (op 534) Assign VGPR 1:U64(677) END
 // (op 534) NOP(680) BEGIN
 // (op 680) generate({})
 // (op 680) end: generate({})
 // (op 534) NOP(680) END
 // (op 534) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}(689) BEGIN
 // (op 689) Assign dim(545) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}
 // tag 545: v**UNALLOCATED**
 // (op 689) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32} into DataFlowTag545: VGPR Value: UInt32 x 1: (unallocated)
 // (op 689) reg expression
 // (op 689) reg expression
 // (op 689) reg expression
 // (op 689) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 689) convert
// Allocated : 1 VGPR (Value: UInt32) (op 689): v21
v_lshrrev_b32 v21, 1, v1 // (op 689) 
 // (op 689) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 689): v22
v_lshl_add_u32 v22, s16, 5, v21 // (op 689) 
 // (op 689) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 689): v21

 // (op 689) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 689): v21
v_and_b32 v21, -2, v1 // (op 689) 
 // (op 689) BEGIN: Flatten[1]TileTile
 // (op 689) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v21:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 689): v23
v_sub_u32 v23, v1, v21 // (op 689) 
 // (op 689) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 689): v21

 // (op 689) BEGIN: Split
// Allocated DataFlowTag545: 1 VGPR (Value: UInt32) (op 689): v21
v_mul_lo_u32 v21, v22, s20 // (op 689) 
v_add_u32 v21, v21, v23 // (op 689) 
 // (op 689) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 689): v23

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 689): v22

 // (op 534) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32}(689) END
 // (op 534) Assign VGPR 2:U64(690) BEGIN
 // (op 690) Assign dim(546) = 2:U64
 // (op 690) Assign stride expression
 // (op 534) Assign VGPR 2:U64(690) END
 // (op 534) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(691) BEGIN
 // (op 691) Assign dim(535) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 691) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // (op 691) reg expression
 // (op 691) reg expression
 // (op 691) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 691) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 691)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 691)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 691): s0, s1
s_mul_i32 s1, s20, s7 // (op 691) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 691): s14
s_mul_i32 s14, s21, s6 // (op 691) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 691) least significant
// Allocated : 1 SGPR (Value: Int32) (op 691): s15
s_mul_hi_u32 s15, s20, s6 // (op 691) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 691) most significant: sum
s_add_u32 s1, s1, s15 // (op 691) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 691): s15

 // Freeing : 1 SGPR (Value: Int32) (op 691): s14

 // (op 691) reg expression
 // (op 691) reg expression
s_mov_b32 s48, s22 // (op 691) reg expression
s_mov_b32 s49, s23 // (op 691) reg expression
s_mov_b32 s50, s0 // (op 691) reg expression
s_mov_b32 s51, 131072 // (op 691) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 691): s0, s1

 // (op 534) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(691) END
 // (op 534) NOP(682) BEGIN
 // (op 682) generate({})
 // (op 682) end: generate({})
 // (op 534) NOP(682) END
 // (op 534) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(696) BEGIN
 // (op 696) Assign dim(548) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 696) Assign stride expression
 // (op 534) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(696) END
 // (op 534) NOP(683) BEGIN
 // (op 683) generate({})
 // (op 683) end: generate({})
 // (op 534) NOP(683) END
 // (op 534) Assign VGPR 1:U64(699) BEGIN
 // (op 699) Assign dim(550) = 1:U64
 // (op 699) Assign stride expression
 // (op 534) Assign VGPR 1:U64(699) END
 // (op 534) NOP(724) BEGIN
 // (op 724) generate({})
 // (op 724) end: generate({})
 // (op 534) NOP(724) END
 // (op 534) Assign VGPR 2:U64(731) BEGIN
 // (op 731) Assign dim(559) = 2:U64
 // (op 731) Assign stride expression
 // (op 534) Assign VGPR 2:U64(731) END
 // (op 534) NOP(725) BEGIN
 // (op 725) generate({})
 // (op 725) end: generate({})
 // (op 534) NOP(725) END
 // (op 534) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(734) BEGIN
 // (op 734) Assign dim(560) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 560: v**UNALLOCATED**
 // (op 734) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag560: VGPR Value: UInt32 x 1: (unallocated)
 // (op 734) reg expression
 // (op 734) reg expression
 // (op 734) reg expression
 // (op 734) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
 // (op 734) convert
// Allocated : 1 VGPR (Value: UInt32) (op 734): v22
v_and_b32 v22, -2, v1 // (op 734) 
 // (op 734) BEGIN: Flatten[1]TileTile
 // (op 734) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v22:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 734): v23
v_sub_u32 v23, v1, v22 // (op 734) 
 // (op 734) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 734): v22

 // (op 734) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 734): v22
v_lshrrev_b32 v22, 1, v1 // (op 734) 
 // (op 734) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 734): v24
v_lshl_add_u32 v24, s17, 5, v22 // (op 734) 
 // (op 734) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 734): v22

 // (op 734) Multiply({Tile: v24:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 734): v22
v_mul_lo_u32 v22, v24, s24 // (op 734) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 734): v24

 // (op 734) BEGIN: Split
 // (op 734) {Split: Add({Flatten[1]TileTile: v23:U32}, v22:U32)U32}
// Allocated DataFlowTag560: 1 VGPR (Value: UInt32) (op 734): v24
v_add_u32 v24, v23, v22 // (op 734) 
 // (op 734) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 734): v22

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 734): v23

 // (op 534) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(734) END
 // (op 534) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(735) BEGIN
 // (op 735) Assign dim(561) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 735) Assign stride expression
 // (op 534) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(735) END
 // (op 534) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(736) BEGIN
 // (op 736) Assign dim(554) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 554: s**UNALLOCATED**
 // (op 736) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag554: SGPR Buffer: None x 1: (unallocated)
 // (op 736) reg expression
 // (op 736) reg expression
 // (op 736) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 736) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 736)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 736)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 736): s0, s1
s_mul_i32 s1, s24, s13 // (op 736) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 736): s14
s_mul_i32 s14, s25, s12 // (op 736) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 736) least significant
// Allocated : 1 SGPR (Value: Int32) (op 736): s15
s_mul_hi_u32 s15, s24, s12 // (op 736) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 736) most significant: sum
s_add_u32 s1, s1, s15 // (op 736) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 736): s15

 // Freeing : 1 SGPR (Value: Int32) (op 736): s14

 // (op 736) reg expression
 // (op 736) reg expression
// Allocated DataFlowTag554: 4 SGPRs (Buffer: None) (op 736): s52, s53, s54, s55
s_mov_b32 s52, s26 // (op 736) reg expression
s_mov_b32 s53, s27 // (op 736) reg expression
s_mov_b32 s54, s0 // (op 736) reg expression
s_mov_b32 s55, 131072 // (op 736) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 736): s0, s1

 // (op 534) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(736) END
 // (op 534) NOP(726) BEGIN
 // (op 726) generate({})
 // (op 726) end: generate({})
 // (op 534) NOP(726) END
 // (op 534) Assign VGPR 1:U64(741) BEGIN
 // (op 741) Assign dim(563) = 1:U64
 // (op 741) Assign stride expression
 // (op 534) Assign VGPR 1:U64(741) END
 // (op 534) NOP(744) BEGIN
 // (op 744) generate({})
 // (op 744) end: generate({})
 // (op 534) NOP(744) END
 // (op 534) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(753) BEGIN
 // (op 753) Assign dim(564) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 564: v**UNALLOCATED**
 // (op 753) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag564: VGPR Value: UInt32 x 1: (unallocated)
 // (op 753) reg expression
 // (op 753) reg expression
 // (op 753) reg expression
 // (op 753) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
 // (op 753) convert
// Allocated : 1 VGPR (Value: UInt32) (op 753): v22
v_and_b32 v22, -2, v1 // (op 753) 
 // (op 753) BEGIN: Flatten[1]TileTile
 // (op 753) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v22:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 753): v23
v_sub_u32 v23, v1, v22 // (op 753) 
 // (op 753) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 753): v22

 // (op 753) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 753): v22
v_lshrrev_b32 v22, 1, v1 // (op 753) 
 // (op 753) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 753): v25
v_lshl_add_u32 v25, s17, 5, v22 // (op 753) 
 // (op 753) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 753): v22

 // (op 753) Multiply({Tile: v25:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 753): v22
v_mul_lo_u32 v22, v25, s24 // (op 753) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 753): v25

 // (op 753) BEGIN: Split
 // (op 753) {Split: Add({Flatten[1]TileTile: v23:U32}, v22:U32)U32}
// Allocated DataFlowTag564: 1 VGPR (Value: UInt32) (op 753): v25
v_add_u32 v25, v23, v22 // (op 753) 
 // (op 753) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 753): v22

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 753): v23

 // (op 534) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(753) END
 // (op 534) Assign VGPR 2:U64(754) BEGIN
 // (op 754) Assign dim(565) = 2:U64
 // (op 754) Assign stride expression
 // (op 534) Assign VGPR 2:U64(754) END
 // (op 534) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(755) BEGIN
 // (op 755) Assign dim(554) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 755) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // (op 755) reg expression
 // (op 755) reg expression
 // (op 755) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 755) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 755)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 755)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 755): s0, s1
s_mul_i32 s1, s24, s13 // (op 755) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 755): s14
s_mul_i32 s14, s25, s12 // (op 755) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 755) least significant
// Allocated : 1 SGPR (Value: Int32) (op 755): s15
s_mul_hi_u32 s15, s24, s12 // (op 755) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 755) most significant: sum
s_add_u32 s1, s1, s15 // (op 755) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 755): s15

 // Freeing : 1 SGPR (Value: Int32) (op 755): s14

 // (op 755) reg expression
 // (op 755) reg expression
s_mov_b32 s52, s26 // (op 755) reg expression
s_mov_b32 s53, s27 // (op 755) reg expression
s_mov_b32 s54, s0 // (op 755) reg expression
s_mov_b32 s55, 131072 // (op 755) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 755): s0, s1

 // (op 534) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(755) END
 // (op 534) NOP(746) BEGIN
 // (op 746) generate({})
 // (op 746) end: generate({})
 // (op 534) NOP(746) END
 // (op 534) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(760) BEGIN
 // (op 760) Assign dim(567) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 760) Assign stride expression
 // (op 534) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(760) END
 // (op 534) NOP(747) BEGIN
 // (op 747) generate({})
 // (op 747) end: generate({})
 // (op 534) NOP(747) END
 // (op 534) Assign VGPR 1:U64(763) BEGIN
 // (op 763) Assign dim(569) = 1:U64
 // (op 763) Assign stride expression
 // (op 534) Assign VGPR 1:U64(763) END
 // (op 534) NOP(364) BEGIN
 // (op 364) generate({})
 // (op 364) end: generate({})
 // (op 534) NOP(364) END
 // (op 534) SetCoordinate(369) BEGIN
 // (op 369) generate({})
 // (op 369) end: generate({})
 // (op 369) generate({867})
 // (op 369) Scope(867) BEGIN
 // (op 867) Lock Scope 867
 // Lock Branch
 // (op 867) generate({864})
 // (op 867) NOP(864) BEGIN
 // (op 864) generate({})
 // (op 864) end: generate({})
 // (op 867) NOP(864) END
 // (op 867) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(871) BEGIN
 // (op 871) Assign dim(600) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 600: s**UNALLOCATED**
 // (op 871) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag600: SGPR Value: UInt32 x 1: (unallocated)
 // (op 871) reg expression
 // (op 871) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 871): v22
v_and_b32 v22, -4, v1 // (op 871) 
 // (op 871) BEGIN: Tile[1]
 // (op 871) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v22:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 871): v23
v_sub_u32 v23, v1, v22 // (op 871) 
 // (op 871) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 871): v22

 // (op 871) BEGIN: Flatten
 // (op 871) {Flatten: ShiftL({Tile[1]: v23:U32}, 4:U32)U32}
 // (op 871) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 871)         value (Tile[1]: VGPR Value: UInt32 x 1: v23) 
 // (op 871)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 871): v22
v_lshlrev_b32 v22, 4, v23 // (op 871) 
 // (op 871) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 871): v23

 // (op 871) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 871): v23
v_lshrrev_b32 v23, 2, v1 // (op 871) 
 // (op 871) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 871): v26
v_lshl_add_u32 v26, v23, 6, v22 // (op 871) 
 // (op 871) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 871): v22

 // Freeing : 1 VGPR (Value: UInt32) (op 871): v23

// Allocated DataFlowTag600: 1 SGPR (Value: UInt32) (op 871): s0
s_nop 0
v_readfirstlane_b32 s0, v26 // (op 871) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 871): v26

 // (op 867) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(871) END
 // (op 867) Assign VGPR 1024:U32(872) BEGIN
 // (op 872) Assign dim(601) = 1024:U32
 // (op 872) Assign stride expression
 // (op 867) Assign VGPR 1024:U32(872) END
 // (op 867) NOP(865) BEGIN
 // (op 865) generate({})
 // (op 865) end: generate({})
 // (op 867) NOP(865) END
 // (op 867) Assign VGPR 1:U32(876) BEGIN
 // (op 876) Assign dim(603) = 1:U32
 // (op 876) Assign stride expression
 // (op 867) Assign VGPR 1:U32(876) END
 // (op 867) LoadTileDirect2LDS Value: FP8x4(491) BEGIN
 // (op 491) GEN: LoadTileDirect2LDS
 // (op 491) GEN: loadMacroTileDirect2LDS OP 491 LDS 25 MacroTile 9
 // (op 491) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 491) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride523: VGPR Value: UInt64 x 1: (unallocated)
 // (op 491) reg expression
 // (op 491) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 491) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 491)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 491)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 491): s14, s15
s_lshl_b64 s[14:15], s[8:9], 4 // (op 491) 
// Allocated Stride523: 2 VGPRs (Value: UInt64) (op 491): v22, v23
v_mov_b64 v[22:23], s[14:15] // (op 491) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 491): s14, s15

 // (op 491) Generate 1:U64 into nullptr
 // (op 491) Generate 1024:U32 into nullptr
 // (op 491) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride523: VGPR Value: UInt64 x 1: v[22:23]
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 491): v26, v27
v_mov_b64 v[26:27], v[12:13] // (op 491) 
 // (op 491) Generate Add(0:U32, {DataFlowTag600Flatten: s0:U32})U32 into nullptr
 // (op 491)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride523: v[22:23]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 491) Generate {DataFlowTag600Flatten: s0:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s0 // (op 491) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 491) Load value direct to lds
// VMEM: Expected complete at 798 (current 318)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 491) Generate Add(v[26:27]:U64, v22:R)U64 into VGPR Value: UInt64 x 1: v[26:27]
 // (op 491) reg expression
 // (op 491) reg expression
 // (op 491) Add(v[26:27]:U64, v22:R)U64
 // (op 491) Add: dest (VGPR Value: UInt64 x 1: v[26:27]) = 
 // (op 491)      lhs (VGPR Value: UInt64 x 1: v[26:27]) 
 // (op 491)      rhs (VGPR Value: Raw32 x 1: v22)
// Allocated : 2 SGPRs (Value: Bool64) (op 491): s14, s15
v_add_co_u32 v26, s[14:15], v26, v22 // (op 491) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 491) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 491): s14, s15

 // (op 491) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 491) reg expression
 // (op 491) Add(m0:U32, 1024:U32)U32
 // (op 491) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 491) 
 // (op 491) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 491) Load value direct to lds
// VMEM: Expected complete at 802 (current 322)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // Freeing Stride523: 2 VGPRs (Value: UInt64) (op 491): v22, v23

 // Freeing : 2 VGPRs (Value: UInt64) (op 491): v26, v27

 // (op 867) LoadTileDirect2LDS Value: FP8x4(491) END
 // (op 867) Deallocate{}(1381) BEGIN
 // (op 1381) Deallocate 602
 // Deleting tag 602
 // (op 1381) Deallocate 600
 // Freeing DataFlowTag600: 1 SGPR (Value: UInt32) (op 871): s0

 // Deleting tag 600
 // (op 1381) Deallocate 601
 // Deleting tag 601
 // (op 1381) Deallocate 603
 // Deleting tag 603
 // (op 867) Deallocate{}(1381) END
 // (op 867) end: generate({864})
 // (op 867) Unlock Scope 867
 // Unlock None
 // (op 369) Scope(867) END
 // (op 369) end: generate({867})
 // (op 534) SetCoordinate(369) END
 // (op 534) SetCoordinate(372) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag560: VGPR Value: UInt32 x 1: v24
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

// Allocated : 1 VGPR (Value: E8M0) (op 373): v22
buffer_load_ubyte v22, v24, s[52:55], 0 offen offset: 0 // (op 373) Load value
// VMEM: Expected complete at 803 (current 323)
 // (op 372) LoadTiled Value: E8M0(373) END
 // (op 372) end: generate({373})
 // (op 534) SetCoordinate(372) END
 // (op 534) SetCoordinate(375) BEGIN
 // (op 375) generate({})
 // (op 375) end: generate({})
 // (op 375) generate({815})
 // (op 375) Scope(815) BEGIN
 // (op 815) Lock Scope 815
 // Lock Branch
 // (op 815) generate({812})
 // (op 815) NOP(812) BEGIN
 // (op 812) generate({})
 // (op 812) end: generate({})
 // (op 815) NOP(812) END
 // (op 815) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(819) BEGIN
 // (op 819) Assign dim(584) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 584: s**UNALLOCATED**
 // (op 819) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag584: SGPR Value: UInt32 x 1: (unallocated)
 // (op 819) reg expression
 // (op 819) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 819): v23
v_and_b32 v23, -4, v1 // (op 819) 
 // (op 819) BEGIN: Tile[1]
 // (op 819) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v23:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 819): v26
v_sub_u32 v26, v1, v23 // (op 819) 
 // (op 819) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 819): v23

 // (op 819) BEGIN: Flatten
 // (op 819) {Flatten: ShiftL({Tile[1]: v26:U32}, 4:U32)U32}
 // (op 819) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 819)         value (Tile[1]: VGPR Value: UInt32 x 1: v26) 
 // (op 819)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 819): v23
v_lshlrev_b32 v23, 4, v26 // (op 819) 
 // (op 819) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 819): v26

 // (op 819) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 819): v26
v_lshrrev_b32 v26, 2, v1 // (op 819) 
 // (op 819) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 819): v27
v_lshl_add_u32 v27, v26, 6, v23 // (op 819) 
 // (op 819) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 819): v23

 // Freeing : 1 VGPR (Value: UInt32) (op 819): v26

// Allocated DataFlowTag584: 1 SGPR (Value: UInt32) (op 819): s0
s_nop 0
v_readfirstlane_b32 s0, v27 // (op 819) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 819): v27

 // (op 815) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(819) END
 // (op 815) Assign VGPR 1024:U32(820) BEGIN
 // (op 820) Assign dim(585) = 1024:U32
 // (op 820) Assign stride expression
 // (op 815) Assign VGPR 1024:U32(820) END
 // (op 815) NOP(813) BEGIN
 // (op 813) generate({})
 // (op 813) end: generate({})
 // (op 815) NOP(813) END
 // (op 815) Assign VGPR 1:U32(824) BEGIN
 // (op 824) Assign dim(587) = 1:U32
 // (op 824) Assign stride expression
 // (op 815) Assign VGPR 1:U32(824) END
 // (op 815) LoadTileDirect2LDS Value: BF8x4(496) BEGIN
 // (op 496) GEN: LoadTileDirect2LDS
 // (op 496) GEN: loadMacroTileDirect2LDS OP 496 LDS 17 MacroTile 8
 // (op 496) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 496) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride504: VGPR Value: UInt64 x 1: (unallocated)
 // (op 496) reg expression
 // (op 496) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 496) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 496)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 496)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 496): s14, s15
s_lshl_b64 s[14:15], s[2:3], 4 // (op 496) 
// Allocated Stride504: 2 VGPRs (Value: UInt64) (op 496): v26, v27
v_mov_b64 v[26:27], s[14:15] // (op 496) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 496): s14, s15

 // (op 496) Generate 1:U64 into nullptr
 // (op 496) Generate 1024:U32 into nullptr
 // (op 496) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 2048
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride504: VGPR Value: UInt64 x 1: v[26:27]
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
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 496): v28, v29
v_mov_b64 v[28:29], v[16:17] // (op 496) 
 // (op 496) Generate Add(2048:U32, {DataFlowTag584Flatten: s0:U32})U32 into nullptr
 // (op 496) reg expression
 // (op 496) Add(2048:U32, {DataFlowTag584Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 496): s1
s_add_u32 s1, 2048, s0 // (op 496) 
 // (op 496)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride504: v[26:27]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 496) Generate s1:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 496) reg expression
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 496) Load value direct to lds
// VMEM: Expected stall of 463. CBNW: 0, Inc: 3
// VMEM: Expected complete at 1279 (current 799)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 496) Generate Add(v[28:29]:U64, v26:R)U64 into VGPR Value: UInt64 x 1: v[28:29]
 // (op 496) reg expression
 // (op 496) reg expression
 // (op 496) Add(v[28:29]:U64, v26:R)U64
 // (op 496) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 496)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 496)      rhs (VGPR Value: Raw32 x 1: v26)
// Allocated : 2 SGPRs (Value: Bool64) (op 496): s14, s15
v_add_co_u32 v28, s[14:15], v28, v26 // (op 496) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 496) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 496): s14, s15

 // (op 496) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 496) reg expression
 // (op 496) Add(m0:U32, 1024:U32)U32
 // (op 496) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 496) 
 // (op 496) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v28, s[40:43], 0 offen offset: 0 lds // (op 496) Load value direct to lds
// VMEM: Expected complete at 1283 (current 803)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // Freeing Stride504: 2 VGPRs (Value: UInt64) (op 496): v26, v27

 // Freeing : 2 VGPRs (Value: UInt64) (op 496): v28, v29

 // Freeing : 1 SGPR (Value: UInt32) (op 496): s1

 // (op 815) LoadTileDirect2LDS Value: BF8x4(496) END
 // (op 815) Deallocate{}(1383) BEGIN
 // (op 1383) Deallocate 585
 // Deleting tag 585
 // (op 1383) Deallocate 587
 // Deleting tag 587
 // (op 1383) Deallocate 584
 // Freeing DataFlowTag584: 1 SGPR (Value: UInt32) (op 819): s0

 // Deleting tag 584
 // (op 1383) Deallocate 586
 // Deleting tag 586
 // (op 815) Deallocate{}(1383) END
 // (op 815) end: generate({812})
 // (op 815) Unlock Scope 815
 // Unlock None
 // (op 375) Scope(815) END
 // (op 375) end: generate({815})
 // (op 534) SetCoordinate(375) END
 // (op 534) SetCoordinate(378) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag541: VGPR Value: UInt32 x 1: v20
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

// Allocated : 1 VGPR (Value: E8M0) (op 379): v23
buffer_load_ubyte v23, v20, s[48:51], 0 offen offset: 0 // (op 379) Load value
// VMEM: Expected complete at 1284 (current 804)
 // (op 378) LoadTiled Value: E8M0(379) END
 // (op 378) end: generate({379})
 // (op 534) SetCoordinate(378) END
 // (op 534) SetCoordinate(381) BEGIN
 // (op 381) generate({})
 // (op 381) end: generate({})
 // (op 381) generate({1085})
 // (op 381) Scope(1085) BEGIN
 // (op 1085) Lock Scope 1085
 // Lock Branch
 // (op 1085) generate({1082})
 // (op 1085) NOP(1082) BEGIN
 // (op 1082) generate({})
 // (op 1082) end: generate({})
 // (op 1085) NOP(1082) END
 // (op 1085) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1089) BEGIN
 // (op 1089) Assign dim(666) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 666: s**UNALLOCATED**
 // (op 1089) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag666: SGPR Value: UInt32 x 1: (unallocated)
 // (op 1089) reg expression
 // (op 1089) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1089): v26
v_and_b32 v26, -4, v1 // (op 1089) 
 // (op 1089) BEGIN: Tile[1]
 // (op 1089) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v26:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1089): v27
v_sub_u32 v27, v1, v26 // (op 1089) 
 // (op 1089) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1089): v26

 // (op 1089) BEGIN: Flatten
 // (op 1089) {Flatten: ShiftL({Tile[1]: v27:U32}, 4:U32)U32}
 // (op 1089) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1089)         value (Tile[1]: VGPR Value: UInt32 x 1: v27) 
 // (op 1089)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1089): v26
v_lshlrev_b32 v26, 4, v27 // (op 1089) 
 // (op 1089) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1089): v27

 // (op 1089) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1089): v27
v_lshrrev_b32 v27, 2, v1 // (op 1089) 
 // (op 1089) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1089): v28
v_lshl_add_u32 v28, v27, 6, v26 // (op 1089) 
 // (op 1089) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1089): v26

 // Freeing : 1 VGPR (Value: UInt32) (op 1089): v27

// Allocated DataFlowTag666: 1 SGPR (Value: UInt32) (op 1089): s0
s_nop 0
v_readfirstlane_b32 s0, v28 // (op 1089) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1089): v28

 // (op 1085) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1089) END
 // (op 1085) Assign VGPR 1024:U32(1090) BEGIN
 // (op 1090) Assign dim(667) = 1024:U32
 // (op 1090) Assign stride expression
 // (op 1085) Assign VGPR 1024:U32(1090) END
 // (op 1085) NOP(1083) BEGIN
 // (op 1083) generate({})
 // (op 1083) end: generate({})
 // (op 1085) NOP(1083) END
 // (op 1085) Assign VGPR 1:U32(1094) BEGIN
 // (op 1094) Assign dim(669) = 1:U32
 // (op 1094) Assign stride expression
 // (op 1085) Assign VGPR 1:U32(1094) END
 // (op 1085) LoadTileDirect2LDS Value: FP8x4(501) BEGIN
 // (op 501) GEN: LoadTileDirect2LDS
 // (op 501) GEN: loadMacroTileDirect2LDS OP 501 LDS 476 MacroTile 474
 // (op 501) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 501) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride523: VGPR Value: UInt64 x 1: (unallocated)
 // (op 501) reg expression
 // (op 501) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 501) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 501)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 501)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 501): s14, s15
s_lshl_b64 s[14:15], s[8:9], 4 // (op 501) 
// Allocated Stride523: 2 VGPRs (Value: UInt64) (op 501): v26, v27
v_mov_b64 v[26:27], s[14:15] // (op 501) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 501): s14, s15

 // (op 501) Generate 1:U64 into nullptr
 // (op 501) Generate 1024:U32 into nullptr
 // (op 501) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 4096
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride523: VGPR Value: UInt64 x 1: v[26:27]
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 501): v28, v29
v_mov_b64 v[28:29], v[12:13] // (op 501) 
 // (op 501) Generate Add(4096:U32, {DataFlowTag666Flatten: s0:U32})U32 into nullptr
 // (op 501) reg expression
 // (op 501) Add(4096:U32, {DataFlowTag666Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 501): s1
s_add_u32 s1, 4096, s0 // (op 501) 
 // (op 501)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride523: v[26:27]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 501) Generate s1:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 501) reg expression
buffer_load_dwordx4 v28, s[44:47], 64 offen offset: 0 lds // (op 501) Load value direct to lds
// VMEM: Expected stall of 463. CBNW: 3, Inc: 3
// VMEM: Expected complete at 1760 (current 1280)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // (op 501) Generate Add(v[28:29]:U64, v26:R)U64 into VGPR Value: UInt64 x 1: v[28:29]
 // (op 501) reg expression
 // (op 501) reg expression
 // (op 501) Add(v[28:29]:U64, v26:R)U64
 // (op 501) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 501)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 501)      rhs (VGPR Value: Raw32 x 1: v26)
// Allocated : 2 SGPRs (Value: Bool64) (op 501): s14, s15
v_add_co_u32 v28, s[14:15], v28, v26 // (op 501) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 501) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 501): s14, s15

 // (op 501) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 501) reg expression
 // (op 501) Add(m0:U32, 1024:U32)U32
 // (op 501) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 501) 
 // (op 501) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v28, s[44:47], 64 offen offset: 0 lds // (op 501) Load value direct to lds
// VMEM: Expected complete at 1764 (current 1284)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // Freeing Stride523: 2 VGPRs (Value: UInt64) (op 501): v26, v27

 // Freeing : 2 VGPRs (Value: UInt64) (op 501): v28, v29

 // Freeing : 1 SGPR (Value: UInt32) (op 501): s1

 // (op 1085) LoadTileDirect2LDS Value: FP8x4(501) END
 // (op 1085) Deallocate{}(1385) BEGIN
 // (op 1385) Deallocate 669
 // Deleting tag 669
 // (op 1385) Deallocate 668
 // Deleting tag 668
 // (op 1385) Deallocate 666
 // Freeing DataFlowTag666: 1 SGPR (Value: UInt32) (op 1089): s0

 // Deleting tag 666
 // (op 1385) Deallocate 667
 // Deleting tag 667
 // (op 1085) Deallocate{}(1385) END
 // (op 1085) end: generate({1082})
 // (op 1085) Unlock Scope 1085
 // Unlock None
 // (op 381) Scope(1085) END
 // (op 381) end: generate({1085})
 // (op 534) SetCoordinate(381) END
 // (op 534) Deallocate{}(1357) BEGIN
 // (op 1357) Deallocate 522
 // Freeing DataFlowTag522: 2 VGPRs (Value: UInt64) (op 606): v12, v13

 // Deleting tag 522
 // (op 1357) Deallocate 524
 // Deleting tag 524
 // (op 1357) Deallocate 521
 // Deleting tag 521
 // (op 1357) Deallocate 525
 // Deleting tag 525
 // (op 1357) Deallocate 523
 // Deleting tag 523
 // (op 534) Deallocate{}(1357) END
 // (op 534) SetCoordinate(384) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag560: VGPR Value: UInt32 x 1: v24
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

// Allocated : 1 VGPR (Value: E8M0) (op 385): v12
buffer_load_ubyte v12, v24, s[52:55], 0 offen offset:2 // (op 385) Load value
// VMEM: Expected complete at 1765 (current 1285)
 // (op 384) LoadTiled Value: E8M0(385) END
 // (op 384) end: generate({385})
 // (op 534) SetCoordinate(384) END
 // (op 534) Deallocate{}(1359) BEGIN
 // (op 1359) Deallocate 562
 // Deleting tag 562
 // (op 1359) Deallocate 559
 // Deleting tag 559
 // (op 1359) Deallocate 561
 // Deleting tag 561
 // (op 1359) Deallocate 563
 // Deleting tag 563
 // (op 1359) Deallocate 560
 // Freeing DataFlowTag560: 1 VGPR (Value: UInt32) (op 734): v24

 // Deleting tag 560
 // (op 534) Deallocate{}(1359) END
 // (op 534) SetCoordinate(387) BEGIN
 // (op 387) generate({})
 // (op 387) end: generate({})
 // (op 387) generate({1046})
 // (op 387) Scope(1046) BEGIN
 // (op 1046) Lock Scope 1046
 // Lock Branch
 // (op 1046) generate({1043})
 // (op 1046) NOP(1043) BEGIN
 // (op 1043) generate({})
 // (op 1043) end: generate({})
 // (op 1046) NOP(1043) END
 // (op 1046) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1050) BEGIN
 // (op 1050) Assign dim(654) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 654: s**UNALLOCATED**
 // (op 1050) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag654: SGPR Value: UInt32 x 1: (unallocated)
 // (op 1050) reg expression
 // (op 1050) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1050): v13
v_and_b32 v13, -4, v1 // (op 1050) 
 // (op 1050) BEGIN: Tile[1]
 // (op 1050) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1050): v24
v_sub_u32 v24, v1, v13 // (op 1050) 
 // (op 1050) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1050): v13

 // (op 1050) BEGIN: Flatten
 // (op 1050) {Flatten: ShiftL({Tile[1]: v24:U32}, 4:U32)U32}
 // (op 1050) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1050)         value (Tile[1]: VGPR Value: UInt32 x 1: v24) 
 // (op 1050)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1050): v13
v_lshlrev_b32 v13, 4, v24 // (op 1050) 
 // (op 1050) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1050): v24

 // (op 1050) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1050): v24
v_lshrrev_b32 v24, 2, v1 // (op 1050) 
 // (op 1050) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1050): v26
v_lshl_add_u32 v26, v24, 6, v13 // (op 1050) 
 // (op 1050) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1050): v13

 // Freeing : 1 VGPR (Value: UInt32) (op 1050): v24

// Allocated DataFlowTag654: 1 SGPR (Value: UInt32) (op 1050): s0
s_nop 0
v_readfirstlane_b32 s0, v26 // (op 1050) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1050): v26

 // (op 1046) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1050) END
 // (op 1046) Assign VGPR 1024:U32(1051) BEGIN
 // (op 1051) Assign dim(655) = 1024:U32
 // (op 1051) Assign stride expression
 // (op 1046) Assign VGPR 1024:U32(1051) END
 // (op 1046) NOP(1044) BEGIN
 // (op 1044) generate({})
 // (op 1044) end: generate({})
 // (op 1046) NOP(1044) END
 // (op 1046) Assign VGPR 1:U32(1055) BEGIN
 // (op 1055) Assign dim(657) = 1:U32
 // (op 1055) Assign stride expression
 // (op 1046) Assign VGPR 1:U32(1055) END
 // (op 1046) LoadTileDirect2LDS Value: BF8x4(503) BEGIN
 // (op 503) GEN: LoadTileDirect2LDS
 // (op 503) GEN: loadMacroTileDirect2LDS OP 503 LDS 472 MacroTile 470
 // (op 503) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 503) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride504: VGPR Value: UInt64 x 1: (unallocated)
 // (op 503) reg expression
 // (op 503) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 503) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 503)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 503)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 503): s14, s15
s_lshl_b64 s[14:15], s[2:3], 4 // (op 503) 
// Allocated Stride504: 2 VGPRs (Value: UInt64) (op 503): v26, v27
v_mov_b64 v[26:27], s[14:15] // (op 503) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 503): s14, s15

 // (op 503) Generate 1:U64 into nullptr
 // (op 503) Generate 1024:U32 into nullptr
 // (op 503) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 6144
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride504: VGPR Value: UInt64 x 1: v[26:27]
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
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 503): v28, v29
v_mov_b64 v[28:29], v[16:17] // (op 503) 
 // (op 503) Generate Add(6144:U32, {DataFlowTag654Flatten: s0:U32})U32 into nullptr
 // (op 503) reg expression
 // (op 503) Add(6144:U32, {DataFlowTag654Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 503): s1
s_add_u32 s1, 6144, s0 // (op 503) 
 // (op 503)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride504: v[26:27]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 503) Generate s1:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 503) reg expression
buffer_load_dwordx4 v28, s[40:43], 64 offen offset: 0 lds // (op 503) Load value direct to lds
// VMEM: Expected stall of 463. CBNW: 6, Inc: 3
// VMEM: Expected complete at 2241 (current 1761)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // (op 503) Generate Add(v[28:29]:U64, v26:R)U64 into VGPR Value: UInt64 x 1: v[28:29]
 // (op 503) reg expression
 // (op 503) reg expression
 // (op 503) Add(v[28:29]:U64, v26:R)U64
 // (op 503) Add: dest (VGPR Value: UInt64 x 1: v[28:29]) = 
 // (op 503)      lhs (VGPR Value: UInt64 x 1: v[28:29]) 
 // (op 503)      rhs (VGPR Value: Raw32 x 1: v26)
// Allocated : 2 SGPRs (Value: Bool64) (op 503): s14, s15
v_add_co_u32 v28, s[14:15], v28, v26 // (op 503) least significant half
v_addc_co_u32 v29, s[14:15], 0, v29, s[14:15] // (op 503) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 503): s14, s15

 // (op 503) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 503) reg expression
 // (op 503) Add(m0:U32, 1024:U32)U32
 // (op 503) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 503) 
 // (op 503) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v28, s[40:43], 64 offen offset: 0 lds // (op 503) Load value direct to lds
// VMEM: Expected complete at 2245 (current 1765)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // Freeing Stride504: 2 VGPRs (Value: UInt64) (op 503): v26, v27

 // Freeing : 2 VGPRs (Value: UInt64) (op 503): v28, v29

 // Freeing : 1 SGPR (Value: UInt32) (op 503): s1

 // (op 1046) LoadTileDirect2LDS Value: BF8x4(503) END
 // (op 1046) Deallocate{}(1387) BEGIN
 // (op 1387) Deallocate 654
 // Freeing DataFlowTag654: 1 SGPR (Value: UInt32) (op 1050): s0

 // Deleting tag 654
 // (op 1387) Deallocate 655
 // Deleting tag 655
 // (op 1387) Deallocate 657
 // Deleting tag 657
 // (op 1387) Deallocate 656
 // Deleting tag 656
 // (op 1046) Deallocate{}(1387) END
 // (op 1046) end: generate({1043})
 // (op 1046) Unlock Scope 1046
 // Unlock None
 // (op 387) Scope(1046) END
 // (op 387) end: generate({1046})
 // (op 534) SetCoordinate(387) END
 // (op 534) Deallocate{}(1361) BEGIN
 // (op 1361) Deallocate 504
 // Deleting tag 504
 // (op 1361) Deallocate 505
 // Deleting tag 505
 // (op 1361) Deallocate 502
 // Deleting tag 502
 // (op 1361) Deallocate 503
 // Freeing DataFlowTag503: 2 VGPRs (Value: UInt64) (op 542): v16, v17

 // Deleting tag 503
 // (op 1361) Deallocate 506
 // Deleting tag 506
 // (op 534) Deallocate{}(1361) END
 // (op 534) SetCoordinate(390) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag541: VGPR Value: UInt32 x 1: v20
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

// Allocated : 1 VGPR (Value: E8M0) (op 391): v13
buffer_load_ubyte v13, v20, s[48:51], 0 offen offset:2 // (op 391) Load value
// VMEM: Expected complete at 2246 (current 1766)
 // (op 390) LoadTiled Value: E8M0(391) END
 // (op 390) end: generate({391})
 // (op 534) SetCoordinate(390) END
 // (op 534) Deallocate{}(1363) BEGIN
 // (op 1363) Deallocate 541
 // Freeing DataFlowTag541: 1 VGPR (Value: UInt32) (op 670): v20

 // Deleting tag 541
 // (op 1363) Deallocate 542
 // Deleting tag 542
 // (op 1363) Deallocate 544
 // Deleting tag 544
 // (op 1363) Deallocate 540
 // Deleting tag 540
 // (op 1363) Deallocate 543
 // Deleting tag 543
 // (op 534) Deallocate{}(1363) END
 // (op 534) NOP(493) BEGIN
 // (op 493) generate({})
 // (op 493) end: generate({})
 // (op 534) NOP(493) END
 // (op 534) Scope(972) BEGIN
 // (op 972) Lock Scope 972
 // Lock Branch
 // (op 972) generate({969})
 // (op 972) NOP(969) BEGIN
 // (op 969) generate({})
 // (op 969) end: generate({})
 // (op 972) NOP(969) END
 // (op 972) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(977) BEGIN
 // (op 977) Assign dim(632) = {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}
 // tag 632: v**UNALLOCATED**
 // (op 977) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32} into DataFlowTag632: VGPR Value: UInt32 x 1: (unallocated)
 // (op 977) reg expression
 // (op 977) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 977): v20
v_and_b32 v20, -2, v1 // (op 977) 
 // (op 977) BEGIN: Tile[1]Flatten
 // (op 977) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v20:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 977): v24
v_sub_u32 v24, v1, v20 // (op 977) 
 // (op 977) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 977): v20

 // (op 977) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 977): v20
v_and_b32 v20, 4294967294, v1 // (op 977) 
 // (op 977) BEGIN: Flatten
 // (op 977) {Flatten: Add(v20:U32, {Tile[1]Flatten: v24:U32})U32}
// Allocated DataFlowTag632: 1 VGPR (Value: UInt32) (op 977): v16
v_add_u32 v16, v20, v24 // (op 977) 
 // (op 977) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 977): v24

 // Freeing : 1 VGPR (Value: UInt32) (op 977): v20

 // (op 972) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(977) END
 // (op 972) Assign VGPR 2:U32(978) BEGIN
 // (op 978) Assign dim(633) = 2:U32
 // (op 978) Assign stride expression
 // (op 972) Assign VGPR 2:U32(978) END
 // (op 972) NOP(970) BEGIN
 // (op 970) generate({})
 // (op 970) end: generate({})
 // (op 972) NOP(970) END
 // (op 972) Assign VGPR 1:U32(982) BEGIN
 // (op 982) Assign dim(635) = 1:U32
 // (op 982) Assign stride expression
 // (op 972) Assign VGPR 1:U32(982) END
 // (op 972) StoreLDSTile Value: E8M0(394) BEGIN
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

 // (op 394) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = DataFlowTag632: VGPR Value: UInt32 x 1: v16
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
 // (op 394) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(9)
ds_write_b8 v16, v22 offset:8192 // (op 394) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 1867 (current 1772)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 972) StoreLDSTile Value: E8M0(394) END
 // (op 972) Deallocate{}(1365) BEGIN
 // (op 1365) Deallocate 634
 // Deleting tag 634
 // (op 1365) Deallocate 632
 // Freeing DataFlowTag632: 1 VGPR (Value: UInt32) (op 977): v16

 // Deleting tag 632
 // (op 1365) Deallocate 633
 // Deleting tag 633
 // (op 1365) Deallocate 635
 // Deleting tag 635
 // (op 972) Deallocate{}(1365) END
 // (op 972) end: generate({969})
 // (op 972) Unlock Scope 972
 // Unlock None
 // (op 534) Scope(972) END
 // (op 534) NOP(498) BEGIN
 // (op 498) generate({})
 // (op 498) end: generate({})
 // (op 534) NOP(498) END
 // (op 534) Scope(919) BEGIN
 // (op 919) Lock Scope 919
 // Lock Branch
 // (op 919) generate({916})
 // (op 919) NOP(916) BEGIN
 // (op 916) generate({})
 // (op 916) end: generate({})
 // (op 919) NOP(916) END
 // (op 919) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(924) BEGIN
 // (op 924) Assign dim(616) = {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}
 // tag 616: v**UNALLOCATED**
 // (op 924) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32} into DataFlowTag616: VGPR Value: UInt32 x 1: (unallocated)
 // (op 924) reg expression
 // (op 924) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 924): v20
v_and_b32 v20, -2, v1 // (op 924) 
 // (op 924) BEGIN: Tile[1]Flatten
 // (op 924) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v20:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 924): v24
v_sub_u32 v24, v1, v20 // (op 924) 
 // (op 924) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 924): v20

 // (op 924) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 924): v20
v_and_b32 v20, 4294967294, v1 // (op 924) 
 // (op 924) BEGIN: Flatten
 // (op 924) {Flatten: Add(v20:U32, {Tile[1]Flatten: v24:U32})U32}
// Allocated DataFlowTag616: 1 VGPR (Value: UInt32) (op 924): v16
v_add_u32 v16, v20, v24 // (op 924) 
 // (op 924) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 924): v24

 // Freeing : 1 VGPR (Value: UInt32) (op 924): v20

 // (op 919) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(924) END
 // (op 919) Assign VGPR 2:U32(925) BEGIN
 // (op 925) Assign dim(617) = 2:U32
 // (op 925) Assign stride expression
 // (op 919) Assign VGPR 2:U32(925) END
 // (op 919) NOP(917) BEGIN
 // (op 917) generate({})
 // (op 917) end: generate({})
 // (op 919) NOP(917) END
 // (op 919) Assign VGPR 1:U32(929) BEGIN
 // (op 929) Assign dim(619) = 1:U32
 // (op 929) Assign stride expression
 // (op 919) Assign VGPR 1:U32(929) END
 // (op 919) StoreLDSTile Value: E8M0(396) BEGIN
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
 // 	info.data = VGPR Value: E8M0 x 1: v23
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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
 // 	info.data = VGPR Value: E8M0 x 1: v23
 // 	info.rowOffsetReg = DataFlowTag616: VGPR Value: UInt32 x 1: v16
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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

s_waitcnt vmcnt(6)
ds_write_b8 v16, v23 offset:8256 // (op 396) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 1873 (current 1778)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 919) StoreLDSTile Value: E8M0(396) END
 // (op 919) Deallocate{}(1367) BEGIN
 // (op 1367) Deallocate 616
 // Freeing DataFlowTag616: 1 VGPR (Value: UInt32) (op 924): v16

 // Deleting tag 616
 // (op 1367) Deallocate 618
 // Deleting tag 618
 // (op 1367) Deallocate 617
 // Deleting tag 617
 // (op 1367) Deallocate 619
 // Deleting tag 619
 // (op 919) Deallocate{}(1367) END
 // (op 919) end: generate({916})
 // (op 919) Unlock Scope 919
 // Unlock None
 // (op 534) Scope(919) END
 // (op 534) Barrier(1207) BEGIN
s_barrier  // (op 1207) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 534) Barrier(1207) END
 // (op 534) SetCoordinate(412) BEGIN
 // (op 412) generate({})
 // (op 412) end: generate({})
 // (op 412) generate({413})
 // (op 412) LoadLDSTile{Value: FP8x4}(413) BEGIN
 // (op 413) GEN: loadMacroTileWAVELDS OP 413 LDS 25 WaveTile 171
 // (op 413) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 413) Generate 1:U32 into nullptr
 // tag 23: v**UNALLOCATED**
 // (op 413) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: (unallocated)
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

 // (op 413) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag604: VGPR Value: UInt32 x 1: v4
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
 // (op 413) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

// Allocated : 8 VGPRs (Value: FP8x4) (op 413): v26, v27, v28, v29, v30, v31, v32, v33
ds_read_b128 v[26:29], v4 // (op 413) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 1875 (current 1780)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
ds_read_b128 v[30:33], v4 offset:32 // (op 413) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 87. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 1963 (current 1868)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 412) LoadLDSTile{Value: FP8x4}(413) END
 // (op 412) end: generate({413})
 // (op 534) SetCoordinate(412) END
 // (op 534) Barrier(1241) BEGIN
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1241) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 534) Barrier(1241) END
 // (op 534) SetCoordinate(418) BEGIN
 // (op 418) generate({})
 // (op 418) end: generate({})
 // (op 418) generate({419})
 // (op 418) LoadLDSTile{Value: E8M0}(419) BEGIN
 // (op 419) GEN: loadMacroTileWAVELDS OP 419 LDS 41 WaveTile 297
 // (op 419) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 419) Generate 1:U32 into nullptr
 // tag 39: v**UNALLOCATED**
 // (op 419) 	Dir = Load
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

 // (op 419) LSTInfo {
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
 // (op 419) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 419): v20
ds_read_u8 v20, v6 offset:8192 // (op 419) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected stall of 3. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 1969 (current 1874)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 418) LoadLDSTile{Value: E8M0}(419) END
 // (op 418) end: generate({419})
 // (op 534) SetCoordinate(418) END
 // (op 534) Barrier(1191) BEGIN
s_barrier  // (op 1191) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 534) Barrier(1191) END
 // (op 534) SetCoordinate(409) BEGIN
 // (op 409) generate({})
 // (op 409) end: generate({})
 // (op 409) generate({410})
 // (op 409) LoadLDSTile{Value: BF8x4}(410) BEGIN
 // (op 410) GEN: loadMacroTileWAVELDS OP 410 LDS 17 WaveTile 105
 // (op 410) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 410) Generate 1:U32 into nullptr
 // tag 15: v**UNALLOCATED**
 // (op 410) 	Dir = Load
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

 // (op 410) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag588: VGPR Value: UInt32 x 1: v3
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
 // (op 410) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

// Allocated : 8 VGPRs (Value: BF8x4) (op 410): v34, v35, v36, v37, v38, v39, v40, v41
ds_read_b128 v[34:37], v3 offset:2048 // (op 410) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 1971 (current 1876)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
ds_read_b128 v[38:41], v3 offset:2080 // (op 410) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 87. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 2059 (current 1964)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 409) LoadLDSTile{Value: BF8x4}(410) END
 // (op 409) end: generate({410})
 // (op 534) SetCoordinate(409) END
 // (op 534) Barrier(1224) BEGIN
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1224) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 534) Barrier(1224) END
 // (op 534) SetCoordinate(415) BEGIN
 // (op 415) generate({})
 // (op 415) end: generate({})
 // (op 415) generate({416})
 // (op 415) LoadLDSTile{Value: E8M0}(416) BEGIN
 // (op 416) GEN: loadMacroTileWAVELDS OP 416 LDS 33 WaveTile 237
 // (op 416) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 416) Generate 1:U32 into nullptr
 // tag 31: v**UNALLOCATED**
 // (op 416) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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

 // (op 416) LSTInfo {
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 416) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 416): v24
ds_read_u8 v24, v0 offset:8256 // (op 416) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected stall of 3. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 2065 (current 1970)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 415) LoadLDSTile{Value: E8M0}(416) END
 // (op 415) end: generate({416})
 // (op 534) SetCoordinate(415) END
 // (op 534) NOP(367) BEGIN
 // (op 367) generate({})
 // (op 367) end: generate({})
 // (op 534) NOP(367) END
 // (op 534) NOP(787) BEGIN
 // (op 787) generate({})
 // (op 787) end: generate({})
 // (op 534) NOP(787) END
 // (op 534) NOP(784) BEGIN
 // (op 784) generate({})
 // (op 784) end: generate({})
 // (op 534) NOP(784) END
 // (op 534) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(791) BEGIN
 // (op 791) Assign dim(576) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 576: s**UNALLOCATED**
 // (op 791) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag576: SGPR Value: UInt32 x 1: (unallocated)
 // (op 791) reg expression
 // (op 791) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 791): v16
v_and_b32 v16, -4, v1 // (op 791) 
 // (op 791) BEGIN: Tile[1]
 // (op 791) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 791): v17
v_sub_u32 v17, v1, v16 // (op 791) 
 // (op 791) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 791): v16

 // (op 791) BEGIN: Flatten
 // (op 791) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 791) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 791)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 791)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 791): v16
v_lshlrev_b32 v16, 4, v17 // (op 791) 
 // (op 791) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 791): v17

 // (op 791) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 791): v17
v_lshrrev_b32 v17, 2, v1 // (op 791) 
 // (op 791) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 791): v42
v_lshl_add_u32 v42, v17, 6, v16 // (op 791) 
 // (op 791) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 791): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 791): v17

// Allocated DataFlowTag576: 1 SGPR (Value: UInt32) (op 791): s0
s_nop 0
v_readfirstlane_b32 s0, v42 // (op 791) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 791): v42

 // (op 534) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(791) END
 // (op 534) Assign VGPR 1024:U32(792) BEGIN
 // (op 792) Assign dim(577) = 1024:U32
 // (op 792) Assign stride expression
 // (op 534) Assign VGPR 1024:U32(792) END
 // (op 534) NOP(785) BEGIN
 // (op 785) generate({})
 // (op 785) end: generate({})
 // (op 534) NOP(785) END
 // (op 534) Assign VGPR 1:U32(796) BEGIN
 // (op 796) Assign dim(579) = 1:U32
 // (op 796) Assign stride expression
 // (op 534) Assign VGPR 1:U32(796) END
 // (op 534) NOP(838) BEGIN
 // (op 838) generate({})
 // (op 838) end: generate({})
 // (op 534) NOP(838) END
 // (op 534) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(843) BEGIN
 // (op 843) Assign dim(592) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 592: s**UNALLOCATED**
 // (op 843) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag592: SGPR Value: UInt32 x 1: (unallocated)
 // (op 843) reg expression
 // (op 843) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 843): v16
v_and_b32 v16, -4, v1 // (op 843) 
 // (op 843) BEGIN: Tile[1]
 // (op 843) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 843): v17
v_sub_u32 v17, v1, v16 // (op 843) 
 // (op 843) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 843): v16

 // (op 843) BEGIN: Flatten
 // (op 843) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 843) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 843)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 843)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 843): v16
v_lshlrev_b32 v16, 4, v17 // (op 843) 
 // (op 843) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 843): v17

 // (op 843) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 843): v17
v_lshrrev_b32 v17, 2, v1 // (op 843) 
 // (op 843) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 843): v42
v_lshl_add_u32 v42, v17, 6, v16 // (op 843) 
 // (op 843) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 843): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 843): v17

// Allocated DataFlowTag592: 1 SGPR (Value: UInt32) (op 843): s1
s_nop 0
v_readfirstlane_b32 s1, v42 // (op 843) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 843): v42

 // (op 534) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(843) END
 // (op 534) Assign VGPR 1024:U32(844) BEGIN
 // (op 844) Assign dim(593) = 1024:U32
 // (op 844) Assign stride expression
 // (op 534) Assign VGPR 1024:U32(844) END
 // (op 534) NOP(839) BEGIN
 // (op 839) generate({})
 // (op 839) end: generate({})
 // (op 534) NOP(839) END
 // (op 534) Assign VGPR 1:U32(848) BEGIN
 // (op 848) Assign dim(595) = 1:U32
 // (op 848) Assign stride expression
 // (op 534) Assign VGPR 1:U32(848) END
 // (op 534) NOP(890) BEGIN
 // (op 890) generate({})
 // (op 890) end: generate({})
 // (op 534) NOP(890) END
 // (op 534) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(895) BEGIN
 // (op 895) Assign dim(608) = {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}
 // tag 608: v**UNALLOCATED**
 // (op 895) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32} into DataFlowTag608: VGPR Value: UInt32 x 1: (unallocated)
 // (op 895) reg expression
 // (op 895) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 895): v16
v_and_b32 v16, -2, v1 // (op 895) 
 // (op 895) BEGIN: Tile[1]Flatten
 // (op 895) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 895): v17
v_sub_u32 v17, v1, v16 // (op 895) 
 // (op 895) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 895): v16

 // (op 895) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 895): v16
v_and_b32 v16, 4294967294, v1 // (op 895) 
 // (op 895) BEGIN: Flatten
 // (op 895) {Flatten: Add(v16:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag608: 1 VGPR (Value: UInt32) (op 895): v42
v_add_u32 v42, v16, v17 // (op 895) 
 // (op 895) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 895): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 895): v16

 // (op 534) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(895) END
 // (op 534) Assign VGPR 2:U32(896) BEGIN
 // (op 896) Assign dim(609) = 2:U32
 // (op 896) Assign stride expression
 // (op 534) Assign VGPR 2:U32(896) END
 // (op 534) NOP(891) BEGIN
 // (op 891) generate({})
 // (op 891) end: generate({})
 // (op 534) NOP(891) END
 // (op 534) Assign VGPR 1:U32(900) BEGIN
 // (op 900) Assign dim(611) = 1:U32
 // (op 900) Assign stride expression
 // (op 534) Assign VGPR 1:U32(900) END
 // (op 534) NOP(943) BEGIN
 // (op 943) generate({})
 // (op 943) end: generate({})
 // (op 534) NOP(943) END
 // (op 534) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(948) BEGIN
 // (op 948) Assign dim(624) = {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}
 // tag 624: v**UNALLOCATED**
 // (op 948) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32} into DataFlowTag624: VGPR Value: UInt32 x 1: (unallocated)
 // (op 948) reg expression
 // (op 948) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 948): v16
v_and_b32 v16, -2, v1 // (op 948) 
 // (op 948) BEGIN: Tile[1]Flatten
 // (op 948) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 948): v17
v_sub_u32 v17, v1, v16 // (op 948) 
 // (op 948) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 948): v16

 // (op 948) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 948): v16
v_and_b32 v16, 4294967294, v1 // (op 948) 
 // (op 948) BEGIN: Flatten
 // (op 948) {Flatten: Add(v16:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag624: 1 VGPR (Value: UInt32) (op 948): v43
v_add_u32 v43, v16, v17 // (op 948) 
 // (op 948) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 948): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 948): v16

 // (op 534) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(948) END
 // (op 534) Assign VGPR 2:U32(949) BEGIN
 // (op 949) Assign dim(625) = 2:U32
 // (op 949) Assign stride expression
 // (op 534) Assign VGPR 2:U32(949) END
 // (op 534) NOP(944) BEGIN
 // (op 944) generate({})
 // (op 944) end: generate({})
 // (op 534) NOP(944) END
 // (op 534) Assign VGPR 1:U32(953) BEGIN
 // (op 953) Assign dim(627) = 1:U32
 // (op 953) Assign stride expression
 // (op 534) Assign VGPR 1:U32(953) END
 // (op 534) NOP(1030) BEGIN
 // (op 1030) generate({})
 // (op 1030) end: generate({})
 // (op 534) NOP(1030) END
 // (op 534) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1035) BEGIN
 // (op 1035) Assign dim(650) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 650: s**UNALLOCATED**
 // (op 1035) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag650: SGPR Value: UInt32 x 1: (unallocated)
 // (op 1035) reg expression
 // (op 1035) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1035): v16
v_and_b32 v16, -4, v1 // (op 1035) 
 // (op 1035) BEGIN: Tile[1]
 // (op 1035) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1035): v17
v_sub_u32 v17, v1, v16 // (op 1035) 
 // (op 1035) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1035): v16

 // (op 1035) BEGIN: Flatten
 // (op 1035) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 1035) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1035)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 1035)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1035): v16
v_lshlrev_b32 v16, 4, v17 // (op 1035) 
 // (op 1035) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1035): v17

 // (op 1035) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1035): v17
v_lshrrev_b32 v17, 2, v1 // (op 1035) 
 // (op 1035) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1035): v44
v_lshl_add_u32 v44, v17, 6, v16 // (op 1035) 
 // (op 1035) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1035): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 1035): v17

// Allocated DataFlowTag650: 1 SGPR (Value: UInt32) (op 1035): s14
s_nop 0
v_readfirstlane_b32 s14, v44 // (op 1035) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1035): v44

 // (op 534) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1035) END
 // (op 534) Assign VGPR 1024:U32(1036) BEGIN
 // (op 1036) Assign dim(651) = 1024:U32
 // (op 1036) Assign stride expression
 // (op 534) Assign VGPR 1024:U32(1036) END
 // (op 534) NOP(1031) BEGIN
 // (op 1031) generate({})
 // (op 1031) end: generate({})
 // (op 534) NOP(1031) END
 // (op 534) Assign VGPR 1:U32(1040) BEGIN
 // (op 1040) Assign dim(653) = 1:U32
 // (op 1040) Assign stride expression
 // (op 534) Assign VGPR 1:U32(1040) END
 // (op 534) NOP(1069) BEGIN
 // (op 1069) generate({})
 // (op 1069) end: generate({})
 // (op 534) NOP(1069) END
 // (op 534) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1074) BEGIN
 // (op 1074) Assign dim(662) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 662: s**UNALLOCATED**
 // (op 1074) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag662: SGPR Value: UInt32 x 1: (unallocated)
 // (op 1074) reg expression
 // (op 1074) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1074): v16
v_and_b32 v16, -4, v1 // (op 1074) 
 // (op 1074) BEGIN: Tile[1]
 // (op 1074) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1074): v17
v_sub_u32 v17, v1, v16 // (op 1074) 
 // (op 1074) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1074): v16

 // (op 1074) BEGIN: Flatten
 // (op 1074) {Flatten: ShiftL({Tile[1]: v17:U32}, 4:U32)U32}
 // (op 1074) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1074)         value (Tile[1]: VGPR Value: UInt32 x 1: v17) 
 // (op 1074)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 1074): v16
v_lshlrev_b32 v16, 4, v17 // (op 1074) 
 // (op 1074) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1074): v17

 // (op 1074) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1074): v17
v_lshrrev_b32 v17, 2, v1 // (op 1074) 
 // (op 1074) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1074): v44
v_lshl_add_u32 v44, v17, 6, v16 // (op 1074) 
 // (op 1074) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1074): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 1074): v17

// Allocated DataFlowTag662: 1 SGPR (Value: UInt32) (op 1074): s15
s_nop 0
v_readfirstlane_b32 s15, v44 // (op 1074) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1074): v44

 // (op 534) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(1074) END
 // (op 534) Assign VGPR 1024:U32(1075) BEGIN
 // (op 1075) Assign dim(663) = 1024:U32
 // (op 1075) Assign stride expression
 // (op 534) Assign VGPR 1024:U32(1075) END
 // (op 534) NOP(1070) BEGIN
 // (op 1070) generate({})
 // (op 1070) end: generate({})
 // (op 534) NOP(1070) END
 // (op 534) Assign VGPR 1:U32(1079) BEGIN
 // (op 1079) Assign dim(665) = 1:U32
 // (op 1079) Assign stride expression
 // (op 534) Assign VGPR 1:U32(1079) END
 // (op 534) NOP(1108) BEGIN
 // (op 1108) generate({})
 // (op 1108) end: generate({})
 // (op 534) NOP(1108) END
 // (op 534) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(1113) BEGIN
 // (op 1113) Assign dim(674) = {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}
 // tag 674: v**UNALLOCATED**
 // (op 1113) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32} into DataFlowTag674: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1113) reg expression
 // (op 1113) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1113): v16
v_and_b32 v16, -2, v1 // (op 1113) 
 // (op 1113) BEGIN: Tile[1]Flatten
 // (op 1113) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1113): v17
v_sub_u32 v17, v1, v16 // (op 1113) 
 // (op 1113) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1113): v16

 // (op 1113) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1113): v16
v_and_b32 v16, 4294967294, v1 // (op 1113) 
 // (op 1113) BEGIN: Flatten
 // (op 1113) {Flatten: Add(v16:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag674: 1 VGPR (Value: UInt32) (op 1113): v44
v_add_u32 v44, v16, v17 // (op 1113) 
 // (op 1113) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 1113): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 1113): v16

 // (op 534) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(1113) END
 // (op 534) Assign VGPR 2:U32(1114) BEGIN
 // (op 1114) Assign dim(675) = 2:U32
 // (op 1114) Assign stride expression
 // (op 534) Assign VGPR 2:U32(1114) END
 // (op 534) NOP(1109) BEGIN
 // (op 1109) generate({})
 // (op 1109) end: generate({})
 // (op 534) NOP(1109) END
 // (op 534) Assign VGPR 1:U32(1118) BEGIN
 // (op 1118) Assign dim(677) = 1:U32
 // (op 1118) Assign stride expression
 // (op 534) Assign VGPR 1:U32(1118) END
 // (op 534) NOP(1132) BEGIN
 // (op 1132) generate({})
 // (op 1132) end: generate({})
 // (op 534) NOP(1132) END
 // (op 534) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(1137) BEGIN
 // (op 1137) Assign dim(682) = {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}
 // tag 682: v**UNALLOCATED**
 // (op 1137) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32} into DataFlowTag682: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1137) reg expression
 // (op 1137) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1137): v16
v_and_b32 v16, -2, v1 // (op 1137) 
 // (op 1137) BEGIN: Tile[1]Flatten
 // (op 1137) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1137): v17
v_sub_u32 v17, v1, v16 // (op 1137) 
 // (op 1137) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1137): v16

 // (op 1137) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1137): v16
v_and_b32 v16, 4294967294, v1 // (op 1137) 
 // (op 1137) BEGIN: Flatten
 // (op 1137) {Flatten: Add(v16:U32, {Tile[1]Flatten: v17:U32})U32}
// Allocated DataFlowTag682: 1 VGPR (Value: UInt32) (op 1137): v45
v_add_u32 v45, v16, v17 // (op 1137) 
 // (op 1137) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 1137): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 1137): v16

 // (op 534) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(1137) END
 // (op 534) Assign VGPR 2:U32(1138) BEGIN
 // (op 1138) Assign dim(683) = 2:U32
 // (op 1138) Assign stride expression
 // (op 534) Assign VGPR 2:U32(1138) END
 // (op 534) NOP(1133) BEGIN
 // (op 1133) generate({})
 // (op 1133) end: generate({})
 // (op 534) NOP(1133) END
 // (op 534) Assign VGPR 1:U32(1142) BEGIN
 // (op 1142) Assign dim(685) = 1:U32
 // (op 1142) Assign stride expression
 // (op 534) Assign VGPR 1:U32(1142) END
 // (op 534) ForLoopOp KLoop: LessThan(DataFlowTag(397)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL(60) BEGIN
 // (op 60) Initialize For Loop
 // (op 60) generate({61})
 // (op 60) Assign SGPR 0:U32(61) BEGIN
 // (op 61) Assign dim(397) = 0:U32
 // tag 397: s**UNALLOCATED**
 // (op 61) Generate 0:U32 into DataFlowTag397: SGPR Value: UInt32 x 1: (unallocated)
// Allocated DataFlowTag397: 1 SGPR (Value: UInt32) (op 61): s18
s_mov_b32 s18, 0 // (op 61) call()
 // (op 60) Assign SGPR 0:U32(61) END
 // (op 60) end: generate({61})
 // (op 60) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 60) 
 // (op 60) Generate LessThan(DataFlowTag(397)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 60) LessThan({DataFlowTag397: s18:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 60) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32
 // (op 60) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 60): s38, s39
s_ashr_i64 s[38:39], s[28:29], 6 // (op 60) 
// Allocated : 1 SGPR (Value: UInt32) (op 60): s19
s_mov_b32 s19, s38 // (op 60) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 60): s38, s39

// Allocated : 1 SGPR (Value: UInt32) (op 60): s38
s_and_b32 s38, s19, 4294967294 // (op 60) 
 // Freeing : 1 SGPR (Value: UInt32) (op 60): s19

s_cmp_lt_u32 s18, s38 // (op 60) 
 // Freeing : 1 SGPR (Value: UInt32) (op 60): s38

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 60) 
s_cbranch_scc0 RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_60 // (op 60) Condition: Top (jump to RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_60 if false)
RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60:
 // (op 60) 
 // (op 60) generate({62, 90, 1210})
 // (op 60) Assign SGPR Add(DataFlowTag(397)U32, 2:U32)U32(62) BEGIN
 // (op 62) Assign dim(397) = Add(DataFlowTag(397)U32, 2:U32)U32
 // (op 62) Generate Add(DataFlowTag(397)U32, 2:U32)U32 into DataFlowTag397: SGPR Value: UInt32 x 1: s18
 // (op 62) reg expression
 // (op 62) Add({DataFlowTag397: s18:U32}, 2:U32)U32
 // (op 60) Multiply(90) BEGIN
 // (op 90) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile4xFP8, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)S into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
 // (op 60) Barrier(1210) BEGIN
s_add_u32 s18, s18, 2 // (op 62) 
 // (op 60) Assign SGPR Add(DataFlowTag(397)U32, 2:U32)U32(62) END
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[34:41], v[26:33], a[0:15], v24, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 90) 
 // (op 60) Multiply(90) END
s_barrier  // (op 1210) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 60) Barrier(1210) END
 // (op 60) SetCoordinate(292) BEGIN
 // (op 292) generate({})
 // (op 292) end: generate({})
 // (op 292) generate({463})
 // (op 292) LoadTileDirect2LDS Value: FP8x4(463) BEGIN
 // (op 463) GEN: LoadTileDirect2LDS
 // (op 463) GEN: loadMacroTileDirect2LDS OP 463 LDS 25 MacroTile 9
 // (op 463) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 463) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride529: VGPR Value: UInt64 x 1: (unallocated)
 // (op 463) reg expression
 // (op 463) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 463) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 463)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 463)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 463): s38, s39
s_lshl_b64 s[38:39], s[8:9], 4 // (op 463) 
// Allocated Stride529: 2 VGPRs (Value: UInt64) (op 463): v16, v17
v_mov_b64 v[16:17], s[38:39] // (op 463) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 463): s38, s39

 // (op 463) Generate 1:U64 into nullptr
 // (op 463) Generate 1024:U32 into nullptr
 // (op 463) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride529: VGPR Value: UInt64 x 1: v[16:17]
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 528: v**UNALLOCATED**
// Allocated Offset528: 2 VGPRs (Value: UInt64) (op 463): v46, v47
v_mov_b64 v[46:47], v[14:15] // (op 463) 
// Allocated : 2 VGPRs (Value: UInt64) (op 463): v48, v49
v_mov_b64 v[48:49], v[46:47] // (op 463) 
 // (op 463) Generate Add(0:U32, {DataFlowTag592Flatten: s1:U32})U32 into nullptr
 // (op 463)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride529: v[16:17]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 463) Generate {DataFlowTag592Flatten: s1:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 463) reg expression
 // (op 463) Generate 128:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 128 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 463): s19
s_mov_b32 s19, 128 // (op 463) call()
// 128 is not a supported value!
buffer_load_dwordx4 v48, s[44:47], s19 offen offset: 0 lds // (op 463) Load value direct to lds
// VMEM: Expected stall of 210. CBNW: 3, Inc: 3
// VMEM: Expected complete at 2722 (current 2242)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 463) Generate Add(v[48:49]:U64, v16:R)U64 into VGPR Value: UInt64 x 1: v[48:49]
 // (op 463) reg expression
 // (op 463) reg expression
 // (op 463) Add(v[48:49]:U64, v16:R)U64
 // (op 463) Add: dest (VGPR Value: UInt64 x 1: v[48:49]) = 
 // (op 463)      lhs (VGPR Value: UInt64 x 1: v[48:49]) 
 // (op 463)      rhs (VGPR Value: Raw32 x 1: v16)
// Allocated : 2 SGPRs (Value: Bool64) (op 463): s38, s39
v_add_co_u32 v48, s[38:39], v48, v16 // (op 463) least significant half
v_addc_co_u32 v49, s[38:39], 0, v49, s[38:39] // (op 463) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 463): s38, s39

 // (op 463) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 463) reg expression
 // (op 463) Add(m0:U32, 1024:U32)U32
 // (op 463) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 463) 
 // (op 463) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v48, s[44:47], s19 offen offset: 0 lds // (op 463) Load value direct to lds
// VMEM: Expected complete at 2726 (current 2246)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 463): s19

 // Freeing Stride529: 2 VGPRs (Value: UInt64) (op 463): v16, v17

 // Freeing : 2 VGPRs (Value: UInt64) (op 463): v48, v49

 // (op 292) LoadTileDirect2LDS Value: FP8x4(463) END
 // (op 292) Deallocate{}(1371) BEGIN
 // (op 1371) Deallocate 594
 // Deleting tag 594
 // (op 292) Deallocate{}(1371) END
 // (op 292) end: generate({463})
 // (op 60) SetCoordinate(292) END
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
// Allocated Offset566: 1 VGPR (Value: UInt32) (op 25): v16
v_mov_b32 v16, v25 // (op 25) 
 // (op 25) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = Offset566: VGPR Value: UInt32 x 1: v16
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

buffer_load_ubyte v22, v16, s[52:55], 0 offen offset:4 // (op 25) Load value
// VMEM: Expected complete at 2728 (current 2248)
 // (op 302) LoadTiled Value: E8M0(25) END
 // (op 302) end: generate({25})
 // (op 60) SetCoordinate(302) END
 // (op 60) Barrier(1194) BEGIN
s_barrier  // (op 1194) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 60) Barrier(1194) END
 // (op 60) SetCoordinate(287) BEGIN
 // (op 287) generate({})
 // (op 287) end: generate({})
 // (op 287) generate({458})
 // (op 287) LoadTileDirect2LDS Value: BF8x4(458) BEGIN
 // (op 458) GEN: LoadTileDirect2LDS
 // (op 458) GEN: loadMacroTileDirect2LDS OP 458 LDS 17 MacroTile 8
 // (op 458) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 458) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride510: VGPR Value: UInt64 x 1: (unallocated)
 // (op 458) reg expression
 // (op 458) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 458) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 458)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 458)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 458): s38, s39
s_lshl_b64 s[38:39], s[2:3], 4 // (op 458) 
// Allocated Stride510: 2 VGPRs (Value: UInt64) (op 458): v48, v49
v_mov_b64 v[48:49], s[38:39] // (op 458) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 458): s38, s39

 // (op 458) Generate 1:U64 into nullptr
 // (op 458) Generate 1024:U32 into nullptr
 // (op 458) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 2048
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride510: VGPR Value: UInt64 x 1: v[48:49]
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
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 509: v**UNALLOCATED**
// Allocated Offset509: 2 VGPRs (Value: UInt64) (op 458): v50, v51
v_mov_b64 v[50:51], v[18:19] // (op 458) 
// Allocated : 2 VGPRs (Value: UInt64) (op 458): v52, v53
v_mov_b64 v[52:53], v[50:51] // (op 458) 
 // (op 458) Generate Add(2048:U32, {DataFlowTag576Flatten: s0:U32})U32 into nullptr
 // (op 458) reg expression
 // (op 458) Add(2048:U32, {DataFlowTag576Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 458): s19
s_add_u32 s19, 2048, s0 // (op 458) 
 // (op 458)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride510: v[48:49]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 458) Generate s19:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s19 // (op 458) reg expression
 // (op 458) Generate 128:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 128 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 458): s38
s_mov_b32 s38, 128 // (op 458) call()
// 128 is not a supported value!
buffer_load_dwordx4 v52, s[40:43], s38 offen offset: 0 lds // (op 458) Load value direct to lds
// VMEM: Expected stall of 466. CBNW: 6, Inc: 3
// VMEM: Expected complete at 3203 (current 2723)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 458) Generate Add(v[52:53]:U64, v48:R)U64 into VGPR Value: UInt64 x 1: v[52:53]
 // (op 458) reg expression
 // (op 458) reg expression
 // (op 458) Add(v[52:53]:U64, v48:R)U64
 // (op 458) Add: dest (VGPR Value: UInt64 x 1: v[52:53]) = 
 // (op 458)      lhs (VGPR Value: UInt64 x 1: v[52:53]) 
 // (op 458)      rhs (VGPR Value: Raw32 x 1: v48)
// Allocated : 2 SGPRs (Value: Bool64) (op 458): s56, s57
v_add_co_u32 v52, s[56:57], v52, v48 // (op 458) least significant half
v_addc_co_u32 v53, s[56:57], 0, v53, s[56:57] // (op 458) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 458): s56, s57

 // (op 458) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 458) reg expression
 // (op 458) Add(m0:U32, 1024:U32)U32
 // (op 458) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 458) 
 // (op 458) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v52, s[40:43], s38 offen offset: 0 lds // (op 458) Load value direct to lds
// VMEM: Expected complete at 3207 (current 2727)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 458): s38

 // Freeing Stride510: 2 VGPRs (Value: UInt64) (op 458): v48, v49

 // Freeing : 2 VGPRs (Value: UInt64) (op 458): v52, v53

 // Freeing : 1 SGPR (Value: UInt32) (op 458): s19

 // (op 287) LoadTileDirect2LDS Value: BF8x4(458) END
 // (op 287) Deallocate{}(1369) BEGIN
 // (op 1369) Deallocate 578
 // Deleting tag 578
 // (op 287) Deallocate{}(1369) END
 // (op 287) end: generate({458})
 // (op 60) SetCoordinate(287) END
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
 // 	info.data = VGPR Value: E8M0 x 1: v23
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
// Allocated Offset547: 1 VGPR (Value: UInt32) (op 17): v17
v_mov_b32 v17, v21 // (op 17) 
 // (op 17) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v23
 // 	info.rowOffsetReg = Offset547: VGPR Value: UInt32 x 1: v17
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

buffer_load_ubyte v23, v17, s[48:51], 0 offen offset:4 // (op 17) Load value
// VMEM: Expected complete at 3209 (current 2729)
 // (op 297) LoadTiled Value: E8M0(17) END
 // (op 297) end: generate({17})
 // (op 60) SetCoordinate(297) END
 // (op 60) NOP(488) BEGIN
 // (op 488) generate({})
 // (op 488) end: generate({})
 // (op 60) NOP(488) END
 // (op 60) Barrier(1280) BEGIN
s_barrier  // (op 1280) 
 // (op 60) Barrier(1280) END
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

 // (op 259) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v12
 // 	info.rowOffsetReg = DataFlowTag682: VGPR Value: UInt32 x 1: v45
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
 // (op 259) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(9)
ds_write_b8 v45, v12 offset:8320 // (op 259) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 2870 (current 2775)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8320, Size: 64)
 // (op 60) StoreLDSTile Value: E8M0(259) END
 // (op 60) Deallocate{}(1343) BEGIN
 // (op 1343) Deallocate 684
 // Deleting tag 684
 // (op 60) Deallocate{}(1343) END
 // (op 60) NOP(483) BEGIN
 // (op 483) generate({})
 // (op 483) end: generate({})
 // (op 60) NOP(483) END
 // (op 60) Barrier(1274) BEGIN
s_barrier  // (op 1274) 
 // (op 60) Barrier(1274) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 8384
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
 // 	info.data = VGPR Value: E8M0 x 1: v13
 // 	info.rowOffsetReg = DataFlowTag674: VGPR Value: UInt32 x 1: v44
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
 // 	info.offset = Literal Value: UInt32 x 0: 8384
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

s_waitcnt vmcnt(6)
ds_write_b8 v44, v13 offset:8384 // (op 258) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 2873 (current 2778)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8384, Size: 64)
 // (op 60) StoreLDSTile Value: E8M0(258) END
 // (op 60) Deallocate{}(1341) BEGIN
 // (op 1341) Deallocate 676
 // Deleting tag 676
 // (op 60) Deallocate{}(1341) END
 // (op 60) Barrier(1264) BEGIN
s_barrier  // (op 1264) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // (op 60) Barrier(1264) END
 // (op 60) SetCoordinate(250) BEGIN
 // (op 250) generate({})
 // (op 250) end: generate({})
 // (op 250) generate({251})
 // (op 250) LoadLDSTile{Value: FP8x4}(251) BEGIN
 // (op 251) GEN: loadMacroTileWAVELDS OP 251 LDS 476 WaveTile 171
 // (op 251) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 251) Generate 1:U32 into nullptr
 // tag 488: v[26:33] alias 488 -> 23
 // (op 251) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: v[26:33]
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

 // (op 251) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: v[26:33]
 // 	info.rowOffsetReg = DataFlowTag670: VGPR Value: UInt32 x 1: v8
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
 // (op 251) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

ds_read_b128 v[26:29], v8 offset:4096 // (op 251) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 2875 (current 2780)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 2048)
ds_read_b128 v[30:33], v8 offset:4128 // (op 251) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 90. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2966 (current 2871)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // (op 250) LoadLDSTile{Value: FP8x4}(251) END
 // (op 250) Deallocate{}(1325) BEGIN
 // (op 1325) Deallocate 672
 // Deleting tag 672
 // (op 250) Deallocate{}(1325) END
 // (op 250) end: generate({251})
 // (op 60) SetCoordinate(250) END
 // (op 60) Barrier(1277) BEGIN
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1277) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8320, Size: 64)
 // (op 60) Barrier(1277) END
 // (op 60) SetCoordinate(254) BEGIN
 // (op 254) generate({})
 // (op 254) end: generate({})
 // (op 254) generate({255})
 // (op 254) LoadLDSTile{Value: E8M0}(255) BEGIN
 // (op 255) GEN: loadMacroTileWAVELDS OP 255 LDS 484 WaveTile 297
 // (op 255) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 255) Generate 1:U32 into nullptr
 // tag 492: v20 alias 492 -> 39
 // (op 255) 	Dir = Load
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

 // (op 255) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
 // 	info.rowOffsetReg = DataFlowTag686: VGPR Value: UInt32 x 1: v10
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
 // (op 255) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_read_u8 v20, v10 offset:8320 // (op 255) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 2969 (current 2874)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8320, Size: 64)
 // (op 254) LoadLDSTile{Value: E8M0}(255) END
 // (op 254) Deallocate{}(1337) BEGIN
 // (op 1337) Deallocate 688
 // Deleting tag 688
 // (op 254) Deallocate{}(1337) END
 // (op 254) end: generate({255})
 // (op 60) SetCoordinate(254) END
 // (op 60) Deallocate{}(1333) BEGIN
 // (op 1333) Deallocate 484
 // Deleting tag 484
 // (op 60) Deallocate{}(1333) END
 // (op 60) Barrier(1258) BEGIN
 // (op 60) Barrier(1335) BEGIN
s_barrier  // (op 1258) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // (op 60) Barrier(1258) END
s_barrier  // (op 1335) 
 // (op 60) Barrier(1335) END
 // (op 60) SetCoordinate(248) BEGIN
 // (op 248) generate({})
 // (op 248) end: generate({})
 // (op 248) generate({249})
 // (op 248) LoadLDSTile{Value: BF8x4}(249) BEGIN
 // (op 249) GEN: loadMacroTileWAVELDS OP 249 LDS 472 WaveTile 105
 // (op 249) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 249) Generate 1:U32 into nullptr
 // tag 486: v[34:41] alias 486 -> 15
 // (op 249) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[34:41]
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
 // 	info.offset = Literal Value: UInt32 x 0: 6144
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
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[34:41]
 // 	info.rowOffsetReg = DataFlowTag658: VGPR Value: UInt32 x 1: v7
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
 // 	info.offset = Literal Value: UInt32 x 0: 6144
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
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

ds_read_b128 v[34:37], v7 offset:6144 // (op 249) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 2972 (current 2877)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
ds_read_b128 v[38:41], v7 offset:6176 // (op 249) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 89. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 3062 (current 2967)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // (op 248) LoadLDSTile{Value: BF8x4}(249) END
 // (op 248) Deallocate{}(1323) BEGIN
 // (op 1323) Deallocate 660
 // Deleting tag 660
 // (op 248) Deallocate{}(1323) END
 // (op 248) end: generate({249})
 // (op 60) SetCoordinate(248) END
 // (op 60) Barrier(1271) BEGIN
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1271) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8384, Size: 64)
 // (op 60) Barrier(1271) END
 // (op 60) SetCoordinate(252) BEGIN
 // (op 252) generate({})
 // (op 252) end: generate({})
 // (op 252) generate({253})
 // (op 252) LoadLDSTile{Value: E8M0}(253) BEGIN
 // (op 253) GEN: loadMacroTileWAVELDS OP 253 LDS 480 WaveTile 237
 // (op 253) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 253) Generate 1:U32 into nullptr
 // tag 490: v24 alias 490 -> 31
 // (op 253) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v24
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
 // 	info.offset = Literal Value: UInt32 x 0: 8384
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
 // 	info.data = VGPR Value: E8M0 x 1: v24
 // 	info.rowOffsetReg = DataFlowTag678: VGPR Value: UInt32 x 1: v5
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
 // 	info.offset = Literal Value: UInt32 x 0: 8384
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

ds_read_u8 v24, v5 offset:8384 // (op 253) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 3065 (current 2970)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8384, Size: 64)
 // (op 252) LoadLDSTile{Value: E8M0}(253) END
 // (op 252) Deallocate{}(1331) BEGIN
 // (op 1331) Deallocate 680
 // Deleting tag 680
 // (op 252) Deallocate{}(1331) END
 // (op 252) end: generate({253})
 // (op 60) SetCoordinate(252) END
 // (op 60) Deallocate{}(1327) BEGIN
 // (op 1327) Deallocate 480
 // Deleting tag 480
 // (op 60) Deallocate{}(1327) END
 // (op 60) NOP(426) BEGIN
 // (op 426) generate({})
 // (op 426) end: generate({})
 // (op 60) NOP(426) END
 // (op 60) Barrier(1329) BEGIN
 // (op 60) Multiply(256) BEGIN
 // (op 256) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile4xFP8, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)S into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
 // (op 60) Barrier(1268) BEGIN
s_barrier  // (op 1329) 
 // (op 60) Barrier(1329) END
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[34:41], v[26:33], a[0:15], v24, v20 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 256) 
 // (op 60) Multiply(256) END
s_barrier  // (op 1268) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // (op 60) Barrier(1268) END
 // (op 60) Deallocate{}(1339) BEGIN
 // (op 1339) Deallocate 490
 // Deleting tag 490alias (-> 31)
 // (op 1339) Deallocate 486
 // Deleting tag 486alias (-> 15)
 // (op 1339) Deallocate 492
 // Deleting tag 492alias (-> 39)
 // (op 1339) Deallocate 488
 // Deleting tag 488alias (-> 23)
 // (op 60) Deallocate{}(1339) END
 // (op 60) SetCoordinate(313) BEGIN
 // (op 313) generate({})
 // (op 313) end: generate({})
 // (op 313) generate({486})
 // (op 313) LoadTileDirect2LDS Value: FP8x4(486) BEGIN
 // (op 486) GEN: LoadTileDirect2LDS
 // (op 486) GEN: loadMacroTileDirect2LDS OP 486 LDS 476 MacroTile 474
 // (op 486) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 486) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride529: VGPR Value: UInt64 x 1: (unallocated)
 // (op 486) reg expression
 // (op 486) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 486) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 486)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 486)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 486): s38, s39
s_lshl_b64 s[38:39], s[8:9], 4 // (op 486) 
// Allocated Stride529: 2 VGPRs (Value: UInt64) (op 486): v48, v49
v_mov_b64 v[48:49], s[38:39] // (op 486) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 486): s38, s39

 // (op 486) Generate 1:U64 into nullptr
 // (op 486) Generate 1024:U32 into nullptr
 // (op 486) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 4096
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride529: VGPR Value: UInt64 x 1: v[48:49]
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 486): v52, v53
v_mov_b64 v[52:53], v[46:47] // (op 486) 
 // (op 486) Generate Add(4096:U32, {DataFlowTag662Flatten: s15:U32})U32 into nullptr
 // (op 486) reg expression
 // (op 486) Add(4096:U32, {DataFlowTag662Flatten: s15:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 486): s19
s_add_u32 s19, 4096, s15 // (op 486) 
 // (op 486)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride529: v[48:49]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 486) Generate s19:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s19 // (op 486) reg expression
 // (op 486) Generate 192:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 192 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 486): s38
s_mov_b32 s38, 192 // (op 486) call()
// 192 is not a supported value!
buffer_load_dwordx4 v52, s[44:47], s38 offen offset: 0 lds // (op 486) Load value direct to lds
// VMEM: Expected stall of 266. CBNW: 3, Inc: 3
// VMEM: Expected complete at 3684 (current 3204)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // (op 486) Generate Add(v[52:53]:U64, v48:R)U64 into VGPR Value: UInt64 x 1: v[52:53]
 // (op 486) reg expression
 // (op 486) reg expression
 // (op 486) Add(v[52:53]:U64, v48:R)U64
 // (op 486) Add: dest (VGPR Value: UInt64 x 1: v[52:53]) = 
 // (op 486)      lhs (VGPR Value: UInt64 x 1: v[52:53]) 
 // (op 486)      rhs (VGPR Value: Raw32 x 1: v48)
// Allocated : 2 SGPRs (Value: Bool64) (op 486): s56, s57
v_add_co_u32 v52, s[56:57], v52, v48 // (op 486) least significant half
v_addc_co_u32 v53, s[56:57], 0, v53, s[56:57] // (op 486) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 486): s56, s57

 // (op 486) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 486) reg expression
 // (op 486) Add(m0:U32, 1024:U32)U32
 // (op 486) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 486) 
 // (op 486) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v52, s[44:47], s38 offen offset: 0 lds // (op 486) Load value direct to lds
// VMEM: Expected complete at 3688 (current 3208)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 4096, Size: 2048)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 486): s38

 // Freeing Stride529: 2 VGPRs (Value: UInt64) (op 486): v48, v49

 // Freeing : 2 VGPRs (Value: UInt64) (op 486): v52, v53

 // Freeing : 1 SGPR (Value: UInt32) (op 486): s19

 // (op 313) LoadTileDirect2LDS Value: FP8x4(486) END
 // (op 313) Deallocate{}(1379) BEGIN
 // (op 1379) Deallocate 664
 // Deleting tag 664
 // (op 313) Deallocate{}(1379) END
 // (op 313) end: generate({486})
 // (op 60) SetCoordinate(313) END
 // (op 60) Deallocate{}(1347) BEGIN
 // (op 1347) Deallocate 530
 // Deleting tag 530
 // (op 1347) Deallocate 528
 // Freeing Offset528: 2 VGPRs (Value: UInt64) (op 463): v46, v47

 // Deleting tag 528
 // (op 60) Deallocate{}(1347) END
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
 // 	info.data = VGPR Value: E8M0 x 1: v12
 // 	info.rowOffsetReg = Offset566: VGPR Value: UInt32 x 1: v16
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

 // (op 60) Assign VGPR Add(DataFlowTag(526)U64, ShiftL(DataFlowTag(527)U64, 1:U32)U64)U64(617) BEGIN
 // (op 617) Assign dim(526) = Add(DataFlowTag(526)U64, ShiftL(DataFlowTag(527)U64, 1:U32)U64)U64
 // (op 617) Generate Add(DataFlowTag(526)U64, ShiftL(DataFlowTag(527)U64, 1:U32)U64)U64 into DataFlowTag526: VGPR Value: UInt64 x 1: v[14:15]
 // (op 617) reg expression
 // (op 617) Add({DataFlowTag526: v[14:15]:U64}, 128:U64)U64
 // (op 617) Add: dest (VGPR Value: UInt64 x 1: v[14:15]) = 
 // (op 617)      lhs (DataFlowTag526: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 617)      rhs (Literal Value: UInt64 x 0: 128)
v_add_co_u32 v14, vcc, 128, v14 // (op 617) least significant half
 // Lock VCC
// Start of Int64 add, locking VCC
buffer_load_ubyte v12, v16, s[52:55], 0 offen offset:6 // (op 262) Load value
// VMEM: Expected complete at 3690 (current 3210)
 // (op 323) LoadTiled Value: E8M0(262) END
 // (op 323) end: generate({262})
 // (op 60) SetCoordinate(323) END
v_addc_co_u32 v15, vcc, 0, v15, vcc // (op 617) most significant half
 // Unlock VCC
// End of Int64 add, Unlocking VCC.
 // (op 60) Assign VGPR Add(DataFlowTag(526)U64, ShiftL(DataFlowTag(527)U64, 1:U32)U64)U64(617) END
 // (op 60) Deallocate{}(1351) BEGIN
 // (op 1351) Deallocate 568
 // Deleting tag 568
 // (op 1351) Deallocate 566
 // Freeing Offset566: 1 VGPR (Value: UInt32) (op 25): v16

 // Deleting tag 566
 // (op 60) Deallocate{}(1351) END
 // (op 60) Assign VGPR Add(DataFlowTag(564)U32, ShiftL(Convert(DataFlowTag(565)U64)U32, 1:U32)U32)U32(745) BEGIN
 // (op 745) Assign dim(564) = Add(DataFlowTag(564)U32, ShiftL(Convert(DataFlowTag(565)U64)U32, 1:U32)U32)U32
 // (op 745) Generate Add(DataFlowTag(564)U32, ShiftL(Convert(DataFlowTag(565)U64)U32, 1:U32)U32)U32 into DataFlowTag564: VGPR Value: UInt32 x 1: v25
 // (op 745) reg expression
 // (op 745) Add({DataFlowTag564: v25:U32}, 4:U32)U32
 // (op 60) Barrier(1261) BEGIN
s_barrier  // (op 1261) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // (op 60) Barrier(1261) END
v_add_u32 v25, 4, v25 // (op 745) 
 // (op 60) Assign VGPR Add(DataFlowTag(564)U32, ShiftL(Convert(DataFlowTag(565)U64)U32, 1:U32)U32)U32(745) END
 // (op 60) SetCoordinate(308) BEGIN
 // (op 308) generate({})
 // (op 308) end: generate({})
 // (op 308) generate({481})
 // (op 308) LoadTileDirect2LDS Value: BF8x4(481) BEGIN
 // (op 481) GEN: LoadTileDirect2LDS
 // (op 481) GEN: loadMacroTileDirect2LDS OP 481 LDS 472 MacroTile 470
 // (op 481) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 481) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride510: VGPR Value: UInt64 x 1: (unallocated)
 // (op 481) reg expression
 // (op 481) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 481) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 481)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 481)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 481): s38, s39
s_lshl_b64 s[38:39], s[2:3], 4 // (op 481) 
// Allocated Stride510: 2 VGPRs (Value: UInt64) (op 481): v46, v47
v_mov_b64 v[46:47], s[38:39] // (op 481) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 481): s38, s39

 // (op 481) Generate 1:U64 into nullptr
 // (op 481) Generate 1024:U32 into nullptr
 // (op 481) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 6144
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride510: VGPR Value: UInt64 x 1: v[46:47]
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
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 2 VGPRs (Value: UInt64) (op 481): v48, v49
v_mov_b64 v[48:49], v[50:51] // (op 481) 
 // (op 481) Generate Add(6144:U32, {DataFlowTag650Flatten: s14:U32})U32 into nullptr
 // (op 481) reg expression
 // (op 481) Add(6144:U32, {DataFlowTag650Flatten: s14:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 481): s19
s_add_u32 s19, 6144, s14 // (op 481) 
 // (op 481)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride510: v[46:47]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 481) Generate s19:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s19 // (op 481) reg expression
 // (op 481) Generate 192:U32 into OffsetD2L: SGPR Value: UInt32 x 1: (unallocated)
// 192 is not a supported value!
// Allocated OffsetD2L: 1 SGPR (Value: UInt32) (op 481): s38
s_mov_b32 s38, 192 // (op 481) call()
// 192 is not a supported value!
buffer_load_dwordx4 v48, s[40:43], s38 offen offset: 0 lds // (op 481) Load value direct to lds
// VMEM: Expected stall of 465. CBNW: 6, Inc: 3
// VMEM: Expected complete at 4165 (current 3685)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // (op 481) Generate Add(v[48:49]:U64, v46:R)U64 into VGPR Value: UInt64 x 1: v[48:49]
 // (op 481) reg expression
 // (op 481) reg expression
 // (op 481) Add(v[48:49]:U64, v46:R)U64
 // (op 481) Add: dest (VGPR Value: UInt64 x 1: v[48:49]) = 
 // (op 481)      lhs (VGPR Value: UInt64 x 1: v[48:49]) 
 // (op 481)      rhs (VGPR Value: Raw32 x 1: v46)
// Allocated : 2 SGPRs (Value: Bool64) (op 481): s56, s57
v_add_co_u32 v48, s[56:57], v48, v46 // (op 481) least significant half
v_addc_co_u32 v49, s[56:57], 0, v49, s[56:57] // (op 481) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 481): s56, s57

 // (op 481) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 481) reg expression
 // (op 481) Add(m0:U32, 1024:U32)U32
 // (op 481) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 481) 
 // (op 481) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v48, s[40:43], s38 offen offset: 0 lds // (op 481) Load value direct to lds
// VMEM: Expected complete at 4169 (current 3689)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 6144, Size: 2048)
 // Freeing OffsetD2L: 1 SGPR (Value: UInt32) (op 481): s38

 // Freeing Stride510: 2 VGPRs (Value: UInt64) (op 481): v46, v47

 // Freeing : 2 VGPRs (Value: UInt64) (op 481): v48, v49

 // Freeing : 1 SGPR (Value: UInt32) (op 481): s19

 // (op 308) LoadTileDirect2LDS Value: BF8x4(481) END
 // (op 308) Deallocate{}(1377) BEGIN
 // (op 1377) Deallocate 652
 // Deleting tag 652
 // (op 308) Deallocate{}(1377) END
 // (op 308) end: generate({481})
 // (op 60) SetCoordinate(308) END
 // (op 60) Deallocate{}(1345) BEGIN
 // (op 1345) Deallocate 511
 // Deleting tag 511
 // (op 1345) Deallocate 509
 // Freeing Offset509: 2 VGPRs (Value: UInt64) (op 458): v50, v51

 // Deleting tag 509
 // (op 60) Deallocate{}(1345) END
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
 // 	info.data = VGPR Value: E8M0 x 1: v13
 // 	info.rowOffsetReg = Offset547: VGPR Value: UInt32 x 1: v17
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

 // (op 60) Assign VGPR Add(DataFlowTag(507)U64, ShiftL(DataFlowTag(508)U64, 1:U32)U64)U64(553) BEGIN
 // (op 553) Assign dim(507) = Add(DataFlowTag(507)U64, ShiftL(DataFlowTag(508)U64, 1:U32)U64)U64
 // (op 553) Generate Add(DataFlowTag(507)U64, ShiftL(DataFlowTag(508)U64, 1:U32)U64)U64 into DataFlowTag507: VGPR Value: UInt64 x 1: v[18:19]
 // (op 553) reg expression
 // (op 553) Add({DataFlowTag507: v[18:19]:U64}, 128:U64)U64
 // (op 553) Add: dest (VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 553)      lhs (DataFlowTag507: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 553)      rhs (Literal Value: UInt64 x 0: 128)
v_add_co_u32 v18, vcc, 128, v18 // (op 553) least significant half
 // Lock VCC
// Start of Int64 add, locking VCC
buffer_load_ubyte v13, v17, s[48:51], 0 offen offset:6 // (op 261) Load value
// VMEM: Expected complete at 4171 (current 3691)
 // (op 318) LoadTiled Value: E8M0(261) END
 // (op 318) end: generate({261})
 // (op 60) SetCoordinate(318) END
v_addc_co_u32 v19, vcc, 0, v19, vcc // (op 553) most significant half
 // Unlock VCC
// End of Int64 add, Unlocking VCC.
 // (op 60) Assign VGPR Add(DataFlowTag(507)U64, ShiftL(DataFlowTag(508)U64, 1:U32)U64)U64(553) END
 // (op 60) Deallocate{}(1349) BEGIN
 // (op 1349) Deallocate 549
 // Deleting tag 549
 // (op 1349) Deallocate 547
 // Freeing Offset547: 1 VGPR (Value: UInt32) (op 17): v17

 // Deleting tag 547
 // (op 60) Deallocate{}(1349) END
 // (op 60) NOP(465) BEGIN
 // (op 465) generate({})
 // (op 465) end: generate({})
 // (op 60) NOP(465) END
 // (op 60) Assign VGPR Add(DataFlowTag(545)U32, ShiftL(Convert(DataFlowTag(546)U64)U32, 1:U32)U32)U32(681) BEGIN
 // (op 681) Assign dim(545) = Add(DataFlowTag(545)U32, ShiftL(Convert(DataFlowTag(546)U64)U32, 1:U32)U32)U32
 // (op 681) Generate Add(DataFlowTag(545)U32, ShiftL(Convert(DataFlowTag(546)U64)U32, 1:U32)U32)U32 into DataFlowTag545: VGPR Value: UInt32 x 1: v21
 // (op 681) reg expression
 // (op 681) Add({DataFlowTag545: v21:U32}, 4:U32)U32
 // (op 60) Barrier(1244) BEGIN
s_barrier  // (op 1244) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 60) Barrier(1244) END
v_add_u32 v21, 4, v21 // (op 681) 
 // (op 60) Assign VGPR Add(DataFlowTag(545)U32, ShiftL(Convert(DataFlowTag(546)U64)U32, 1:U32)U32)U32(681) END
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

 // (op 30) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v22
 // 	info.rowOffsetReg = DataFlowTag624: VGPR Value: UInt32 x 1: v43
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
ds_write_b8 v43, v22 offset:8192 // (op 30) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 3928 (current 3833)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 60) StoreLDSTile Value: E8M0(30) END
 // (op 60) Deallocate{}(1285) BEGIN
 // (op 1285) Deallocate 626
 // Deleting tag 626
 // (op 60) Deallocate{}(1285) END
 // (op 60) NOP(460) BEGIN
 // (op 460) generate({})
 // (op 460) end: generate({})
 // (op 60) NOP(460) END
 // (op 60) Barrier(1227) BEGIN
s_barrier  // (op 1227) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 60) Barrier(1227) END
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
 // 	info.data = VGPR Value: E8M0 x 1: v23
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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
 // 	info.data = VGPR Value: E8M0 x 1: v23
 // 	info.rowOffsetReg = DataFlowTag608: VGPR Value: UInt32 x 1: v42
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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
ds_write_b8 v42, v23 offset:8256 // (op 21) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 3931 (current 3836)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 60) StoreLDSTile Value: E8M0(21) END
 // (op 60) Deallocate{}(1283) BEGIN
 // (op 1283) Deallocate 610
 // Deleting tag 610
 // (op 60) Deallocate{}(1283) END
 // (op 60) Barrier(1216) BEGIN
s_barrier  // (op 1216) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 60) Barrier(1216) END
 // (op 60) SetCoordinate(75) BEGIN
 // (op 75) generate({})
 // (op 75) end: generate({})
 // (op 75) generate({77})
 // (op 75) LoadLDSTile{Value: FP8x4}(77) BEGIN
 // (op 77) GEN: loadMacroTileWAVELDS OP 77 LDS 25 WaveTile 171
 // (op 77) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 77) Generate 1:U32 into nullptr
 // (op 77) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: v[26:33]
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

 // (op 77) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: v[26:33]
 // 	info.rowOffsetReg = DataFlowTag604: VGPR Value: UInt32 x 1: v4
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
 // (op 77) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

ds_read_b128 v[26:29], v4 // (op 77) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 3933 (current 3838)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
ds_read_b128 v[30:33], v4 offset:32 // (op 77) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 90. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 4024 (current 3929)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 75) LoadLDSTile{Value: FP8x4}(77) END
 // (op 75) end: generate({77})
 // (op 60) SetCoordinate(75) END
 // (op 60) Barrier(1251) BEGIN
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1251) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 60) Barrier(1251) END
 // (op 60) SetCoordinate(85) BEGIN
 // (op 85) generate({})
 // (op 85) end: generate({})
 // (op 85) generate({87})
 // (op 85) LoadLDSTile{Value: E8M0}(87) BEGIN
 // (op 87) GEN: loadMacroTileWAVELDS OP 87 LDS 41 WaveTile 297
 // (op 87) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 87) Generate 1:U32 into nullptr
 // (op 87) 	Dir = Load
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

 // (op 87) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v20
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
 // (op 87) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_read_u8 v20, v6 offset:8192 // (op 87) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4027 (current 3932)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 85) LoadLDSTile{Value: E8M0}(87) END
 // (op 85) end: generate({87})
 // (op 60) SetCoordinate(85) END
 // (op 60) Barrier(1200) BEGIN
s_barrier  // (op 1200) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 60) Barrier(1200) END
 // (op 60) SetCoordinate(70) BEGIN
 // (op 70) generate({})
 // (op 70) end: generate({})
 // (op 70) generate({72})
 // (op 70) LoadLDSTile{Value: BF8x4}(72) BEGIN
 // (op 72) GEN: loadMacroTileWAVELDS OP 72 LDS 17 WaveTile 105
 // (op 72) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 72) Generate 1:U32 into nullptr
 // (op 72) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[34:41]
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

 // (op 72) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[34:41]
 // 	info.rowOffsetReg = DataFlowTag588: VGPR Value: UInt32 x 1: v3
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
 // (op 72) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

ds_read_b128 v[34:37], v3 offset:2048 // (op 72) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4029 (current 3934)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
ds_read_b128 v[38:41], v3 offset:2080 // (op 72) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected stall of 90. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 4120 (current 4025)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 70) LoadLDSTile{Value: BF8x4}(72) END
 // (op 70) end: generate({72})
 // (op 60) SetCoordinate(70) END
 // (op 60) Barrier(1234) BEGIN
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1234) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 60) Barrier(1234) END
 // (op 60) SetCoordinate(80) BEGIN
 // (op 80) generate({})
 // (op 80) end: generate({})
 // (op 80) generate({82})
 // (op 80) LoadLDSTile{Value: E8M0}(82) BEGIN
 // (op 82) GEN: loadMacroTileWAVELDS OP 82 LDS 33 WaveTile 237
 // (op 82) 	varTypeInfo.elementBits = 8
 // 	varTypeInfo.packing = 1

 // (op 82) Generate 1:U32 into nullptr
 // (op 82) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v24
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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
 // 	info.data = VGPR Value: E8M0 x 1: v24
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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

ds_read_u8 v24, v0 offset:8256 // (op 82) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4123 (current 4028)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 80) LoadLDSTile{Value: E8M0}(82) END
 // (op 80) end: generate({82})
 // (op 60) SetCoordinate(80) END
 // (op 60) NOP(427) BEGIN
 // (op 427) generate({})
 // (op 427) end: generate({})
 // (op 60) NOP(427) END
 // (op 60) end: generate({62, 90, 1210})
 // (op 60) For Loop Increment
 // (op 60) generate({})
 // (op 60) end: generate({})
 // (op 60) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60 if true)
 // (op 60) Generate LessThan(DataFlowTag(397)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 60) LessThan({DataFlowTag397: s18:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 60) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32
 // (op 60) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 60): s38, s39
s_ashr_i64 s[38:39], s[28:29], 6 // (op 60) 
// Allocated : 1 SGPR (Value: UInt32) (op 60): s19
s_mov_b32 s19, s38 // (op 60) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 60): s38, s39

// Allocated : 1 SGPR (Value: UInt32) (op 60): s38
s_and_b32 s38, s19, 4294967294 // (op 60) 
 // Freeing : 1 SGPR (Value: UInt32) (op 60): s19

s_cmp_lt_u32 s18, s38 // (op 60) 
 // Freeing : 1 SGPR (Value: UInt32) (op 60): s38

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 60) 
s_cbranch_scc1 RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60 // (op 60) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_60 if true)
RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_60:
 // (op 60) 
 // (op 60) Unlock For Loop
 // Unlock None
 // (op 534) ForLoopOp KLoop: LessThan(DataFlowTag(397)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL(60) END
 // (op 534) Deallocate{}(1291) BEGIN
 // (op 1291) Deallocate 546
 // Deleting tag 546
 // (op 1291) Deallocate 39
 // Deleting tag 39
 // (op 1291) Deallocate 15
 // Deleting tag 15
 // (op 1291) Deallocate 11
 // Freeing : 1 VGPR (Value: E8M0) (op 373): v22

 // Deleting tag 11
 // (op 1291) Deallocate 10
 // Freeing : 1 VGPR (Value: E8M0) (op 379): v23

 // Deleting tag 10
 // (op 1291) Deallocate 8
 // Deleting tag 8
 // (op 1291) Deallocate 685
 // Deleting tag 685
 // (op 1291) Deallocate 683
 // Deleting tag 683
 // (op 1291) Deallocate 682
 // Freeing DataFlowTag682: 1 VGPR (Value: UInt32) (op 1137): v45

 // Deleting tag 682
 // (op 1291) Deallocate 677
 // Deleting tag 677
 // (op 1291) Deallocate 675
 // Deleting tag 675
 // (op 1291) Deallocate 674
 // Freeing DataFlowTag674: 1 VGPR (Value: UInt32) (op 1113): v44

 // Deleting tag 674
 // (op 1291) Deallocate 665
 // Deleting tag 665
 // (op 1291) Deallocate 663
 // Deleting tag 663
 // (op 1291) Deallocate 662
 // Freeing DataFlowTag662: 1 SGPR (Value: UInt32) (op 1074): s15

 // Deleting tag 662
 // (op 1291) Deallocate 653
 // Deleting tag 653
 // (op 1291) Deallocate 651
 // Deleting tag 651
 // (op 1291) Deallocate 650
 // Freeing DataFlowTag650: 1 SGPR (Value: UInt32) (op 1035): s14

 // Deleting tag 650
 // (op 1291) Deallocate 627
 // Deleting tag 627
 // (op 1291) Deallocate 625
 // Deleting tag 625
 // (op 1291) Deallocate 624
 // Freeing DataFlowTag624: 1 VGPR (Value: UInt32) (op 948): v43

 // Deleting tag 624
 // (op 1291) Deallocate 611
 // Deleting tag 611
 // (op 1291) Deallocate 609
 // Deleting tag 609
 // (op 1291) Deallocate 608
 // Freeing DataFlowTag608: 1 VGPR (Value: UInt32) (op 895): v42

 // Deleting tag 608
 // (op 1291) Deallocate 595
 // Deleting tag 595
 // (op 1291) Deallocate 593
 // Deleting tag 593
 // (op 1291) Deallocate 23
 // Deleting tag 23
 // (op 1291) Deallocate 482
 // Deleting tag 482
 // (op 1291) Deallocate 472
 // Deleting tag 472
 // (op 1291) Deallocate 474
 // Deleting tag 474
 // (op 1291) Deallocate 548
 // Deleting tag 548
 // (op 1291) Deallocate 476
 // Deleting tag 476
 // (op 1291) Deallocate 529
 // Deleting tag 529
 // (op 1291) Deallocate 478
 // Deleting tag 478
 // (op 1291) Deallocate 510
 // Deleting tag 510
 // (op 1291) Deallocate 531
 // Deleting tag 531
 // (op 1291) Deallocate 592
 // Freeing DataFlowTag592: 1 SGPR (Value: UInt32) (op 843): s1

 // Deleting tag 592
 // (op 1291) Deallocate 507
 // Freeing DataFlowTag507: 2 VGPRs (Value: UInt64) (op 561): v18, v19

 // Deleting tag 507
 // (op 1291) Deallocate 508
 // Deleting tag 508
 // (op 1291) Deallocate 564
 // Freeing DataFlowTag564: 1 VGPR (Value: UInt32) (op 753): v25

 // Deleting tag 564
 // (op 1291) Deallocate 9
 // Deleting tag 9
 // (op 1291) Deallocate 550
 // Deleting tag 550
 // (op 1291) Deallocate 470
 // Deleting tag 470
 // (op 1291) Deallocate 569
 // Deleting tag 569
 // (op 1291) Deallocate 567
 // Deleting tag 567
 // (op 1291) Deallocate 565
 // Deleting tag 565
 // (op 1291) Deallocate 397
 // Freeing DataFlowTag397: 1 SGPR (Value: UInt32) (op 61): s18

 // Deleting tag 397
 // (op 1291) Deallocate 527
 // Deleting tag 527
 // (op 1291) Deallocate 31
 // Deleting tag 31
 // (op 1291) Deallocate 545
 // Freeing DataFlowTag545: 1 VGPR (Value: UInt32) (op 689): v21

 // Deleting tag 545
 // (op 1291) Deallocate 512
 // Deleting tag 512
 // (op 1291) Deallocate 526
 // Freeing DataFlowTag526: 2 VGPRs (Value: UInt64) (op 625): v14, v15

 // Deleting tag 526
 // (op 1291) Deallocate 576
 // Freeing DataFlowTag576: 1 SGPR (Value: UInt32) (op 791): s0

 // Deleting tag 576
 // (op 1291) Deallocate 577
 // Deleting tag 577
 // (op 1291) Deallocate 579
 // Deleting tag 579
 // (op 534) Deallocate{}(1291) END
 // (op 534) Barrier(1293) BEGIN
s_barrier  // (op 1293) 
 // (op 534) Barrier(1293) END
 // (op 534) end: generate({529})
 // (op 534) Unlock Scope 534
 // Unlock None
 // (op 344) Scope(534) END
 // (op 344) SetCoordinate(221) BEGIN
 // (op 221) generate({})
 // (op 221) end: generate({})
 // (op 221) generate({222})
 // (op 221) SetCoordinate(222) BEGIN
 // (op 222) generate({})
 // (op 222) end: generate({})
 // (op 222) generate({511})
 // (op 222) Scope(511) BEGIN
 // (op 511) Lock Scope 511
 // Lock Branch
 // (op 511) generate({505})
 // (op 511) NOP(505) BEGIN
 // (op 505) generate({})
 // (op 505) end: generate({})
 // (op 511) NOP(505) END
 // (op 511) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64}(516) BEGIN
 // (op 516) Assign dim(494) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64}
 // tag 494: v**UNALLOCATED**
 // (op 516) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64} into DataFlowTag494: VGPR Value: UInt64 x 1: (unallocated)
 // (op 516) reg expression
 // (op 516) reg expression
 // (op 516) reg expression
 // (op 516) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
 // (op 516) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 516): v25
v_lshrrev_b32 v25, 2, v1 // (op 516) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 516): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 516) 
 // (op 516) BEGIN: Tile
 // (op 516) convert
// Allocated : 1 VGPR (Value: UInt32) (op 516): v14
v_lshl_add_u32 v14, s16, 5, v25 // (op 516) 
 // (op 516) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 516): v25

 // (op 516) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 516) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 516): s14
s_lshr_b32 s14, s0, 1 // (op 516) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 516): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 516): v25
v_and_b32 v25, -4, v1 // (op 516) 
 // (op 516) reg expression
 // (op 516) BEGIN: Flatten[1]
 // (op 516) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v25:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 516): v15
v_sub_u32 v15, v1, v25 // (op 516) 
 // (op 516) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 516): v25

 // (op 516) BEGIN: Tile
 // (op 516) {Tile: ShiftL({Flatten[1]: v15:U32}, 4:U32)U32}
 // (op 516) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 516)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 516)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 516): v25
v_lshlrev_b32 v25, 4, v15 // (op 516) 
 // (op 516) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 516): v15

 // (op 516) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 516): v15
v_lshl_add_u32 v15, s14, 7, v25 // (op 516) 
 // (op 516) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 516): v25

 // Freeing : 1 SGPR (Value: UInt32) (op 516): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 516): v22, v23
v_mov_b32 v23, 0 // (op 516) convert
v_mov_b32 v22, v15 // (op 516) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 516): v15

 // (op 516) BEGIN: Split
 // (op 516) Multiply: dest (DataFlowTag494: VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 516)           lhs (Tile: VGPR Value: UInt32 x 1: v14) 
 // (op 516)           rhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3])
// Allocated DataFlowTag494: 2 VGPRs (Value: UInt64) (op 516): v18, v19
v_mul_lo_u32 v19, v14, s3 // (op 516) most significant: low of low * high
 // (op 516) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 516): v21
v_mul_hi_u32 v21, v14, s2 // (op 516) most significant: high of low * low
v_mul_lo_u32 v18, v14, s2 // (op 516) least significant: low of low * low
v_add_u32 v19, v19, v21 // (op 516) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 516): v21

 // (op 516) Add: dest (DataFlowTag494: VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 516)      lhs (DataFlowTag494: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 516)      rhs (Tile: VGPR Value: UInt64 x 1: v[22:23])
// Allocated : 2 SGPRs (Value: Bool64) (op 516): s0, s1
v_add_co_u32 v18, s[0:1], v18, v22 // (op 516) least significant half
v_addc_co_u32 v19, s[0:1], v19, v23, s[0:1] // (op 516) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 516): s0, s1

 // (op 516) END: Split
 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 516): v22, v23

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 516): v14

 // (op 511) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(78)U32, 2:I)U32)U32}, Tensor_0_stride_0_0:I64, Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(78)U32, BitwiseAnd(DataFlowTag(78)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64)U64}(516) END
 // (op 511) Assign VGPR 64:U64(517) BEGIN
 // (op 517) Assign dim(495) = 64:U64
 // (op 517) Assign stride expression
 // (op 511) Assign VGPR 64:U64(517) END
 // (op 511) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(518) BEGIN
 // (op 518) Assign dim(497) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 518) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // (op 518) reg expression
 // (op 518) reg expression
 // (op 518) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 518) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 518)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 518)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 518): s0, s1
s_mul_i32 s1, s2, s7 // (op 518) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 518): s14
s_mul_i32 s14, s3, s6 // (op 518) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 518) least significant
// Allocated : 1 SGPR (Value: Int32) (op 518): s15
s_mul_hi_u32 s15, s2, s6 // (op 518) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 518) most significant: sum
s_add_u32 s1, s1, s15 // (op 518) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 518): s15

 // Freeing : 1 SGPR (Value: Int32) (op 518): s14

 // (op 518) reg expression
 // (op 518) reg expression
s_mov_b32 s40, s4 // (op 518) reg expression
s_mov_b32 s41, s5 // (op 518) reg expression
s_mov_b32 s42, s0 // (op 518) reg expression
s_mov_b32 s43, 131072 // (op 518) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 518): s0, s1

 // (op 511) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(518) END
 // (op 511) NOP(507) BEGIN
 // (op 507) generate({})
 // (op 507) end: generate({})
 // (op 511) NOP(507) END
 // (op 511) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(523) BEGIN
 // (op 523) Assign dim(499) = Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64
 // (op 523) Assign stride expression
 // (op 511) Assign VGPR Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64(523) END
 // (op 511) NOP(508) BEGIN
 // (op 508) generate({})
 // (op 508) end: generate({})
 // (op 511) NOP(508) END
 // (op 511) Assign VGPR 1:U64(526) BEGIN
 // (op 526) Assign dim(501) = 1:U64
 // (op 526) Assign stride expression
 // (op 511) Assign VGPR 1:U64(526) END
 // (op 511) NOP(574) BEGIN
 // (op 574) generate({})
 // (op 574) end: generate({})
 // (op 511) NOP(574) END
 // (op 511) Assign VGPR {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(583) BEGIN
 // (op 583) Assign dim(513) = {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}
 // tag 513: v**UNALLOCATED**
 // (op 583) Generate {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64} into DataFlowTag513: VGPR Value: UInt64 x 1: (unallocated)
 // (op 583) reg expression
 // (op 583) reg expression
 // (op 583) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
 // (op 583) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 583): v25
v_and_b32 v25, -4, v1 // (op 583) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 583): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 583) 
 // (op 583) BEGIN: Flatten[1]
 // (op 583) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v25:U32)U32}
 // (op 583) convert
// Allocated : 1 VGPR (Value: UInt32) (op 583): v14
v_sub_u32 v14, v1, v25 // (op 583) 
 // (op 583) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 583): v25

 // (op 583) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 583) BEGIN: Tile
 // (op 583) {Tile: ShiftL({Flatten[1]: v14:U32}, 4:U32)U32}
 // (op 583) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 583)         value (Flatten[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 583)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 SGPR (Value: UInt32) (op 583): s14
s_lshr_b32 s14, s0, 1 // (op 583) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 583): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 583): v25
v_lshlrev_b32 v25, 4, v14 // (op 583) 
 // (op 583) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 583): v14

 // (op 583) reg expression
 // (op 583) BEGIN: Tile
 // (op 583) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 583): v14
v_lshl_add_u32 v14, s14, 7, v25 // (op 583) 
 // (op 583) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 583): v25

 // Freeing : 1 SGPR (Value: UInt32) (op 583): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 583): v16, v17
v_mov_b32 v17, 0 // (op 583) convert
v_mov_b32 v16, v14 // (op 583) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 583): v14

 // (op 583) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 583): v25
v_lshrrev_b32 v25, 2, v1 // (op 583) 
 // (op 583) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 583): v14
v_lshl_add_u32 v14, s17, 5, v25 // (op 583) 
 // (op 583) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 583): v25

 // (op 583) Multiply({Tile: v14:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 583) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 583)           lhs (Tile: VGPR Value: UInt32 x 1: v14) 
 // (op 583)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 583): v22, v23
v_mul_lo_u32 v23, v14, s9 // (op 583) most significant: low of low * high
 // (op 583) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 583): v15
v_mul_hi_u32 v15, v14, s8 // (op 583) most significant: high of low * low
v_mul_lo_u32 v22, v14, s8 // (op 583) least significant: low of low * low
v_add_u32 v23, v23, v15 // (op 583) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 583): v15

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 583): v14

 // (op 583) BEGIN: Split
 // (op 583) {Split: Add({Tile: v[16:17]:U64}, v[22:23]:I64)U64}
 // (op 583) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 583)      lhs (Tile: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 583)      rhs (VGPR Value: Int64 x 1: v[22:23])
// Allocated DataFlowTag513: 2 VGPRs (Value: UInt64) (op 583): v14, v15
// Allocated : 2 SGPRs (Value: Bool64) (op 583): s0, s1
v_add_co_u32 v14, s[0:1], v16, v22 // (op 583) least significant half
v_addc_co_u32 v15, s[0:1], v17, v23, s[0:1] // (op 583) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 583): s0, s1

 // (op 583) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 583): v22, v23

 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 583): v16, v17

 // (op 511) Assign VGPR {Split: Add(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(144)U32, BitwiseAnd(DataFlowTag(144)U32, -4:I)U32)U32}, 4:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(144)U32, 2:I)U32)U32}, Tensor_2_stride_1_3:I64)I64)U64}(583) END
 // (op 511) Assign VGPR 64:U64(584) BEGIN
 // (op 584) Assign dim(514) = 64:U64
 // (op 584) Assign stride expression
 // (op 511) Assign VGPR 64:U64(584) END
 // (op 511) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(585) BEGIN
 // (op 585) Assign dim(516) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 585) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // (op 585) reg expression
 // (op 585) reg expression
 // (op 585) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 585) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 585)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 585)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 585): s0, s1
s_mul_i32 s1, s8, s13 // (op 585) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 585): s14
s_mul_i32 s14, s9, s12 // (op 585) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 585) least significant
// Allocated : 1 SGPR (Value: Int32) (op 585): s15
s_mul_hi_u32 s15, s8, s12 // (op 585) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 585) most significant: sum
s_add_u32 s1, s1, s15 // (op 585) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 585): s15

 // Freeing : 1 SGPR (Value: Int32) (op 585): s14

 // (op 585) reg expression
 // (op 585) reg expression
s_mov_b32 s44, s10 // (op 585) reg expression
s_mov_b32 s45, s11 // (op 585) reg expression
s_mov_b32 s46, s0 // (op 585) reg expression
s_mov_b32 s47, 131072 // (op 585) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 585): s0, s1

 // (op 511) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(585) END
 // (op 511) NOP(576) BEGIN
 // (op 576) generate({})
 // (op 576) end: generate({})
 // (op 511) NOP(576) END
 // (op 511) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(590) BEGIN
 // (op 590) Assign dim(518) = Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64
 // (op 590) Assign stride expression
 // (op 511) Assign VGPR Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64(590) END
 // (op 511) NOP(577) BEGIN
 // (op 577) generate({})
 // (op 577) end: generate({})
 // (op 511) NOP(577) END
 // (op 511) Assign VGPR 1:U64(593) BEGIN
 // (op 593) Assign dim(520) = 1:U64
 // (op 593) Assign stride expression
 // (op 511) Assign VGPR 1:U64(593) END
 // (op 511) NOP(638) BEGIN
 // (op 638) generate({})
 // (op 638) end: generate({})
 // (op 511) NOP(638) END
 // (op 511) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32})U32}(647) BEGIN
 // (op 647) Assign dim(532) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32})U32}
 // tag 532: v**UNALLOCATED**
 // (op 647) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32})U32} into DataFlowTag532: VGPR Value: UInt32 x 1: (unallocated)
 // (op 647) reg expression
 // (op 647) reg expression
 // (op 647) reg expression
 // (op 647) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 647) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 647): v25
v_lshrrev_b32 v25, 1, v1 // (op 647) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 647): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 647) 
 // (op 647) BEGIN: Tile
 // (op 647) convert
// Allocated : 1 VGPR (Value: UInt32) (op 647): v16
v_lshl_add_u32 v16, s16, 5, v25 // (op 647) 
 // (op 647) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 647): v25

 // (op 647) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 647) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 647): s14
s_lshr_b32 s14, s0, 1 // (op 647) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 647): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 647): v17
v_and_b32 v17, -2, v1 // (op 647) 
 // (op 647) reg expression
 // (op 647) BEGIN: Flatten[1]Tile
 // (op 647) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v17:U32)U32}
 // (op 647) convert
// Allocated : 1 VGPR (Value: UInt32) (op 647): v25
v_sub_u32 v25, v1, v17 // (op 647) 
 // (op 647) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 647): v17

 // (op 647) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 647): v17
v_lshl_add_u32 v17, s14, 2, v25 // (op 647) 
 // (op 647) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 647): v25

 // Freeing : 1 SGPR (Value: UInt32) (op 647): s14

 // (op 647) BEGIN: Split
// Allocated DataFlowTag532: 1 VGPR (Value: UInt32) (op 647): v25
v_mul_lo_u32 v25, v16, s20 // (op 647) 
v_add_u32 v25, v25, v17 // (op 647) 
 // (op 647) END: Split
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 647): v17

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 647): v16

 // (op 511) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, LogicalShiftR(DataFlowTag(210)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(210)U32, BitwiseAnd(DataFlowTag(210)U32, -2:I)U32)U32})U32})U32}(647) END
 // (op 511) Assign VGPR 2:U64(648) BEGIN
 // (op 648) Assign dim(533) = 2:U64
 // (op 648) Assign stride expression
 // (op 511) Assign VGPR 2:U64(648) END
 // (op 511) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(649) BEGIN
 // (op 649) Assign dim(535) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 649) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag535: SGPR Buffer: None x 1: s[48:51]
 // (op 649) reg expression
 // (op 649) reg expression
 // (op 649) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 649) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 649)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 649)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 649): s0, s1
s_mul_i32 s1, s20, s7 // (op 649) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 649): s14
s_mul_i32 s14, s21, s6 // (op 649) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 649) least significant
// Allocated : 1 SGPR (Value: Int32) (op 649): s15
s_mul_hi_u32 s15, s20, s6 // (op 649) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 649) most significant: sum
s_add_u32 s1, s1, s15 // (op 649) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 649): s15

 // Freeing : 1 SGPR (Value: Int32) (op 649): s14

 // (op 649) reg expression
 // (op 649) reg expression
s_mov_b32 s48, s22 // (op 649) reg expression
s_mov_b32 s49, s23 // (op 649) reg expression
s_mov_b32 s50, s0 // (op 649) reg expression
s_mov_b32 s51, 131072 // (op 649) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 649): s0, s1

 // (op 511) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(649) END
 // (op 511) NOP(640) BEGIN
 // (op 640) generate({})
 // (op 640) end: generate({})
 // (op 511) NOP(640) END
 // (op 511) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(654) BEGIN
 // (op 654) Assign dim(537) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 654) Assign stride expression
 // (op 511) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(654) END
 // (op 511) NOP(641) BEGIN
 // (op 641) generate({})
 // (op 641) end: generate({})
 // (op 511) NOP(641) END
 // (op 511) Assign VGPR 1:U64(657) BEGIN
 // (op 657) Assign dim(539) = 1:U64
 // (op 657) Assign stride expression
 // (op 511) Assign VGPR 1:U64(657) END
 // (op 511) NOP(702) BEGIN
 // (op 702) generate({})
 // (op 702) end: generate({})
 // (op 511) NOP(702) END
 // (op 511) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(711) BEGIN
 // (op 711) Assign dim(551) = {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 551: v**UNALLOCATED**
 // (op 711) Generate {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag551: VGPR Value: UInt32 x 1: (unallocated)
 // (op 711) reg expression
 // (op 711) reg expression
 // (op 711) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
 // (op 711) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 711): v16
v_and_b32 v16, -2, v1 // (op 711) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 711): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 711) 
 // (op 711) BEGIN: Flatten[1]Tile
 // (op 711) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
 // (op 711) convert
// Allocated : 1 VGPR (Value: UInt32) (op 711): v17
v_sub_u32 v17, v1, v16 // (op 711) 
 // (op 711) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 711): v16

 // (op 711) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 711) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 711): s14
s_lshr_b32 s14, s0, 1 // (op 711) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 711): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 711): v16
v_lshrrev_b32 v16, 1, v1 // (op 711) 
 // (op 711) reg expression
 // (op 711) BEGIN: Tile
 // (op 711) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 711): v21
v_lshl_add_u32 v21, s14, 2, v17 // (op 711) 
 // (op 711) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 711): v17

 // Freeing : 1 SGPR (Value: UInt32) (op 711): s14

 // (op 711) convert
 // (op 711) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 711): v17
v_lshl_add_u32 v17, s17, 5, v16 // (op 711) 
 // (op 711) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 711): v16

 // (op 711) Multiply({Tile: v17:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 711): v16
v_mul_lo_u32 v16, v17, s24 // (op 711) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 711): v17

 // (op 711) BEGIN: Split
 // (op 711) {Split: Add({Tile: v21:U32}, v16:U32)U32}
// Allocated DataFlowTag551: 1 VGPR (Value: UInt32) (op 711): v17
v_add_u32 v17, v21, v16 // (op 711) 
 // (op 711) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 711): v16

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 711): v21

 // (op 511) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(270)U32, BitwiseAnd(DataFlowTag(270)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, LogicalShiftR(DataFlowTag(270)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(711) END
 // (op 511) Assign VGPR 2:U64(712) BEGIN
 // (op 712) Assign dim(552) = 2:U64
 // (op 712) Assign stride expression
 // (op 511) Assign VGPR 2:U64(712) END
 // (op 511) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(713) BEGIN
 // (op 713) Assign dim(554) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 713) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag554: SGPR Buffer: None x 1: s[52:55]
 // (op 713) reg expression
 // (op 713) reg expression
 // (op 713) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 713) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 713)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 713)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 713): s0, s1
s_mul_i32 s1, s24, s13 // (op 713) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 713): s14
s_mul_i32 s14, s25, s12 // (op 713) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 713) least significant
// Allocated : 1 SGPR (Value: Int32) (op 713): s15
s_mul_hi_u32 s15, s24, s12 // (op 713) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 713) most significant: sum
s_add_u32 s1, s1, s15 // (op 713) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 713): s15

 // Freeing : 1 SGPR (Value: Int32) (op 713): s14

 // (op 713) reg expression
 // (op 713) reg expression
s_mov_b32 s52, s26 // (op 713) reg expression
s_mov_b32 s53, s27 // (op 713) reg expression
s_mov_b32 s54, s0 // (op 713) reg expression
s_mov_b32 s55, 131072 // (op 713) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 713): s0, s1

 // (op 511) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(713) END
 // (op 511) NOP(704) BEGIN
 // (op 704) generate({})
 // (op 704) end: generate({})
 // (op 511) NOP(704) END
 // (op 511) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(718) BEGIN
 // (op 718) Assign dim(556) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 718) Assign stride expression
 // (op 511) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(718) END
 // (op 511) NOP(705) BEGIN
 // (op 705) generate({})
 // (op 705) end: generate({})
 // (op 511) NOP(705) END
 // (op 511) Assign VGPR 1:U64(721) BEGIN
 // (op 721) Assign dim(558) = 1:U64
 // (op 721) Assign stride expression
 // (op 511) Assign VGPR 1:U64(721) END
 // (op 511) NOP(799) BEGIN
 // (op 799) generate({})
 // (op 799) end: generate({})
 // (op 511) NOP(799) END
 // (op 511) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(804) BEGIN
 // (op 804) Assign dim(580) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 580: s**UNALLOCATED**
 // (op 804) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag580: SGPR Value: UInt32 x 1: (unallocated)
 // (op 804) reg expression
 // (op 804) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 804): v16
v_and_b32 v16, -4, v1 // (op 804) 
 // (op 804) BEGIN: Tile[1]
 // (op 804) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 804): v21
v_sub_u32 v21, v1, v16 // (op 804) 
 // (op 804) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 804): v16

 // (op 804) BEGIN: Flatten
 // (op 804) {Flatten: ShiftL({Tile[1]: v21:U32}, 4:U32)U32}
 // (op 804) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 804)         value (Tile[1]: VGPR Value: UInt32 x 1: v21) 
 // (op 804)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 804): v16
v_lshlrev_b32 v16, 4, v21 // (op 804) 
 // (op 804) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 804): v21

 // (op 804) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 804): v21
v_lshrrev_b32 v21, 2, v1 // (op 804) 
 // (op 804) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 804): v22
v_lshl_add_u32 v22, v21, 6, v16 // (op 804) 
 // (op 804) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 804): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 804): v21

// Allocated DataFlowTag580: 1 SGPR (Value: UInt32) (op 804): s0
s_nop 0
v_readfirstlane_b32 s0, v22 // (op 804) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 804): v22

 // (op 511) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(93)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(93)U32, BitwiseAnd(DataFlowTag(93)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(804) END
 // (op 511) Assign VGPR 1024:U32(805) BEGIN
 // (op 805) Assign dim(581) = 1024:U32
 // (op 805) Assign stride expression
 // (op 511) Assign VGPR 1024:U32(805) END
 // (op 511) NOP(800) BEGIN
 // (op 800) generate({})
 // (op 800) end: generate({})
 // (op 511) NOP(800) END
 // (op 511) Assign VGPR 1:U32(809) BEGIN
 // (op 809) Assign dim(583) = 1:U32
 // (op 809) Assign stride expression
 // (op 511) Assign VGPR 1:U32(809) END
 // (op 511) NOP(851) BEGIN
 // (op 851) generate({})
 // (op 851) end: generate({})
 // (op 511) NOP(851) END
 // (op 511) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(856) BEGIN
 // (op 856) Assign dim(596) = ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32
 // tag 596: s**UNALLOCATED**
 // (op 856) Generate ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32 into DataFlowTag596: SGPR Value: UInt32 x 1: (unallocated)
 // (op 856) reg expression
 // (op 856) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 856): v16
v_and_b32 v16, -4, v1 // (op 856) 
 // (op 856) BEGIN: Tile[1]
 // (op 856) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 856): v21
v_sub_u32 v21, v1, v16 // (op 856) 
 // (op 856) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 856): v16

 // (op 856) BEGIN: Flatten
 // (op 856) {Flatten: ShiftL({Tile[1]: v21:U32}, 4:U32)U32}
 // (op 856) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 856)         value (Tile[1]: VGPR Value: UInt32 x 1: v21) 
 // (op 856)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 856): v16
v_lshlrev_b32 v16, 4, v21 // (op 856) 
 // (op 856) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 856): v21

 // (op 856) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 856): v21
v_lshrrev_b32 v21, 2, v1 // (op 856) 
 // (op 856) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 856): v22
v_lshl_add_u32 v22, v21, 6, v16 // (op 856) 
 // (op 856) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 856): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 856): v21

// Allocated DataFlowTag596: 1 SGPR (Value: UInt32) (op 856): s1
s_nop 0
v_readfirstlane_b32 s1, v22 // (op 856) Wait state hazard: VALU Write Readlane Hazard
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 856): v22

 // (op 511) Assign SGPR ToScalar({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(159)U32, 2:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(159)U32, BitwiseAnd(DataFlowTag(159)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32(856) END
 // (op 511) Assign VGPR 1024:U32(857) BEGIN
 // (op 857) Assign dim(597) = 1024:U32
 // (op 857) Assign stride expression
 // (op 511) Assign VGPR 1024:U32(857) END
 // (op 511) NOP(852) BEGIN
 // (op 852) generate({})
 // (op 852) end: generate({})
 // (op 511) NOP(852) END
 // (op 511) Assign VGPR 1:U32(861) BEGIN
 // (op 861) Assign dim(599) = 1:U32
 // (op 861) Assign stride expression
 // (op 511) Assign VGPR 1:U32(861) END
 // (op 511) NOP(903) BEGIN
 // (op 903) generate({})
 // (op 903) end: generate({})
 // (op 511) NOP(903) END
 // (op 511) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(908) BEGIN
 // (op 908) Assign dim(612) = {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}
 // tag 612: v**UNALLOCATED**
 // (op 908) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32} into DataFlowTag612: VGPR Value: UInt32 x 1: (unallocated)
 // (op 908) reg expression
 // (op 908) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 908): v16
v_and_b32 v16, -2, v1 // (op 908) 
 // (op 908) BEGIN: Tile[1]Flatten
 // (op 908) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 908): v21
v_sub_u32 v21, v1, v16 // (op 908) 
 // (op 908) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 908): v16

 // (op 908) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 908): v16
v_and_b32 v16, 4294967294, v1 // (op 908) 
 // (op 908) BEGIN: Flatten
 // (op 908) {Flatten: Add(v16:U32, {Tile[1]Flatten: v21:U32})U32}
// Allocated DataFlowTag612: 1 VGPR (Value: UInt32) (op 908): v22
v_add_u32 v22, v16, v21 // (op 908) 
 // (op 908) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 908): v21

 // Freeing : 1 VGPR (Value: UInt32) (op 908): v16

 // (op 511) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(225)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(225)U32, BitwiseAnd(DataFlowTag(225)U32, -2:I)U32)U32})U32}(908) END
 // (op 511) Assign VGPR 2:U32(909) BEGIN
 // (op 909) Assign dim(613) = 2:U32
 // (op 909) Assign stride expression
 // (op 511) Assign VGPR 2:U32(909) END
 // (op 511) NOP(904) BEGIN
 // (op 904) generate({})
 // (op 904) end: generate({})
 // (op 511) NOP(904) END
 // (op 511) Assign VGPR 1:U32(913) BEGIN
 // (op 913) Assign dim(615) = 1:U32
 // (op 913) Assign stride expression
 // (op 511) Assign VGPR 1:U32(913) END
 // (op 511) NOP(956) BEGIN
 // (op 956) generate({})
 // (op 956) end: generate({})
 // (op 511) NOP(956) END
 // (op 511) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(961) BEGIN
 // (op 961) Assign dim(628) = {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}
 // tag 628: v**UNALLOCATED**
 // (op 961) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32} into DataFlowTag628: VGPR Value: UInt32 x 1: (unallocated)
 // (op 961) reg expression
 // (op 961) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 961): v16
v_and_b32 v16, -2, v1 // (op 961) 
 // (op 961) BEGIN: Tile[1]Flatten
 // (op 961) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 961): v21
v_sub_u32 v21, v1, v16 // (op 961) 
 // (op 961) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 961): v16

 // (op 961) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 961): v16
v_and_b32 v16, 4294967294, v1 // (op 961) 
 // (op 961) BEGIN: Flatten
 // (op 961) {Flatten: Add(v16:U32, {Tile[1]Flatten: v21:U32})U32}
// Allocated DataFlowTag628: 1 VGPR (Value: UInt32) (op 961): v23
v_add_u32 v23, v16, v21 // (op 961) 
 // (op 961) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 961): v21

 // Freeing : 1 VGPR (Value: UInt32) (op 961): v16

 // (op 511) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(285)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(285)U32, BitwiseAnd(DataFlowTag(285)U32, -2:I)U32)U32})U32}(961) END
 // (op 511) Assign VGPR 2:U32(962) BEGIN
 // (op 962) Assign dim(629) = 2:U32
 // (op 962) Assign stride expression
 // (op 511) Assign VGPR 2:U32(962) END
 // (op 511) NOP(957) BEGIN
 // (op 957) generate({})
 // (op 957) end: generate({})
 // (op 511) NOP(957) END
 // (op 511) Assign VGPR 1:U32(966) BEGIN
 // (op 966) Assign dim(631) = 1:U32
 // (op 966) Assign stride expression
 // (op 511) Assign VGPR 1:U32(966) END
 // (op 511) ForLoopOp KLoopTail: LessThan(DataFlowTag(452)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL(175) BEGIN
 // (op 175) Initialize For Loop
 // (op 175) generate({176})
 // (op 175) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32(176) BEGIN
 // (op 176) Assign dim(452) = BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32
 // tag 452: s**UNALLOCATED**
 // (op 176) Generate BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32 into DataFlowTag452: SGPR Value: UInt32 x 1: (unallocated)
 // (op 176) reg expression
 // (op 176) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 176): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 176) 
 // (op 176) convert
 // (op 176) BitwiseAnd({ convertForwardSubset: s14:U32}, 4294967294:U32)U32
// Allocated DataFlowTag452: 1 SGPR (Value: UInt32) (op 176): s18
s_and_b32 s18, s14, 4294967294 // (op 176) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 176): s14, s15

 // (op 175) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32(176) END
 // (op 175) end: generate({176})
 // (op 175) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 175) 
 // (op 175) Generate LessThan(DataFlowTag(452)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 175) LessThan({DataFlowTag452: s18:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32)BL
 // (op 175) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 175): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 175) 
// Allocated : 1 SGPR (Value: UInt32) (op 175): s19
s_mov_b32 s19, s14 // (op 175) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 175): s14, s15

s_cmp_lt_u32 s18, s19 // (op 175) 
 // Freeing : 1 SGPR (Value: UInt32) (op 175): s19

 // Freeing : 1 VGPR (Value: E8M0) (op 385): v12

 // Freeing : 1 VGPR (Value: E8M0) (op 391): v13

 // Freeing : 8 VGPRs (Value: FP8x4) (op 413): v26, v27, v28, v29, v30, v31, v32, v33

 // Freeing : 1 VGPR (Value: E8M0) (op 419): v20

 // Freeing : 8 VGPRs (Value: BF8x4) (op 410): v34, v35, v36, v37, v38, v39, v40, v41

 // Freeing : 1 VGPR (Value: E8M0) (op 416): v24

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
 // (op 175) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 175) 
s_cbranch_scc0 RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_175 // (op 175) Condition: Top (jump to RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_175 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
 // (op 175) 
RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175:
 // (op 175) 
 // (op 175) generate({177, 1197})
 // (op 175) Assign SGPR Add(DataFlowTag(452)U32, 1:U32)U32(177) BEGIN
 // (op 177) Assign dim(452) = Add(DataFlowTag(452)U32, 1:U32)U32
 // (op 177) Generate Add(DataFlowTag(452)U32, 1:U32)U32 into DataFlowTag452: SGPR Value: UInt32 x 1: s18
 // (op 177) reg expression
 // (op 177) Add({DataFlowTag452: s18:U32}, 1:U32)U32
 // (op 175) Barrier(1197) BEGIN
s_barrier  // (op 1197) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 175) Barrier(1197) END
s_add_u32 s18, s18, 1 // (op 177) 
 // (op 175) Assign SGPR Add(DataFlowTag(452)U32, 1:U32)U32(177) END
 // (op 175) SetCoordinate(226) BEGIN
 // (op 226) generate({})
 // (op 226) end: generate({})
 // (op 226) generate({468})
 // (op 226) LoadTileDirect2LDS Value: BF8x4(468) BEGIN
 // (op 468) GEN: LoadTileDirect2LDS
 // (op 468) GEN: loadMacroTileDirect2LDS OP 468 LDS 17 MacroTile 454
 // (op 468) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 468) Generate Convert(ShiftL(Tensor_0_stride_0_0:I64, 4:U32)I64)U64 into Stride499: VGPR Value: UInt64 x 1: (unallocated)
 // (op 468) reg expression
 // (op 468) ShiftL({Tensor_0_stride_0_0: s[2:3]:I64}, 4:U32)I64
 // (op 468) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 468)         value (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 468)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 468): s14, s15
s_lshl_b64 s[14:15], s[2:3], 4 // (op 468) 
// Allocated Stride499: 2 VGPRs (Value: UInt64) (op 468): v12, v13
v_mov_b64 v[12:13], s[14:15] // (op 468) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 468): s14, s15

 // (op 468) Generate 1:U64 into nullptr
 // (op 468) Generate 1024:U32 into nullptr
 // (op 468) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 2048
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride499: VGPR Value: UInt64 x 1: v[12:13]
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
 // 	info.bufDesc = DataFlowTag497: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 498: v**UNALLOCATED**
// Allocated Offset498: 2 VGPRs (Value: UInt64) (op 468): v20, v21
v_mov_b64 v[20:21], v[18:19] // (op 468) 
// Allocated : 2 VGPRs (Value: UInt64) (op 468): v26, v27
v_mov_b64 v[26:27], v[20:21] // (op 468) 
 // (op 468) Generate Add(2048:U32, {DataFlowTag580Flatten: s0:U32})U32 into nullptr
 // (op 468) reg expression
 // (op 468) Add(2048:U32, {DataFlowTag580Flatten: s0:U32})U32
// Allocated : 1 SGPR (Value: UInt32) (op 468): s19
s_add_u32 s19, 2048, s0 // (op 468) 
 // (op 468)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride499: v[12:13]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 468) Generate s19:U32 into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s19 // (op 468) reg expression
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 468) Load value direct to lds
// VMEM: Expected complete at 4664 (current 4184)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 468) Generate Add(v[26:27]:U64, v12:R)U64 into VGPR Value: UInt64 x 1: v[26:27]
 // (op 468) reg expression
 // (op 468) reg expression
 // (op 468) Add(v[26:27]:U64, v12:R)U64
 // (op 468) Add: dest (VGPR Value: UInt64 x 1: v[26:27]) = 
 // (op 468)      lhs (VGPR Value: UInt64 x 1: v[26:27]) 
 // (op 468)      rhs (VGPR Value: Raw32 x 1: v12)
// Allocated : 2 SGPRs (Value: Bool64) (op 468): s14, s15
v_add_co_u32 v26, s[14:15], v26, v12 // (op 468) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 468) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 468): s14, s15

 // (op 468) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 468) reg expression
 // (op 468) Add(m0:U32, 1024:U32)U32
 // (op 468) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 468) 
 // (op 468) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v26, s[40:43], 0 offen offset: 0 lds // (op 468) Load value direct to lds
// VMEM: Expected complete at 4668 (current 4188)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // Freeing Stride499: 2 VGPRs (Value: UInt64) (op 468): v12, v13

 // Freeing : 2 VGPRs (Value: UInt64) (op 468): v26, v27

 // Freeing : 1 SGPR (Value: UInt32) (op 468): s19

 // (op 226) LoadTileDirect2LDS Value: BF8x4(468) END
 // (op 226) Deallocate{}(1373) BEGIN
 // (op 1373) Deallocate 500
 // Deleting tag 500
 // (op 1373) Deallocate 582
 // Deleting tag 582
 // (op 1373) Deallocate 498
 // Freeing Offset498: 2 VGPRs (Value: UInt64) (op 468): v20, v21

 // Deleting tag 498
 // (op 1373) Deallocate 454
 // Deleting tag 454
 // (op 226) Deallocate{}(1373) END
 // (op 226) Assign VGPR Add(DataFlowTag(494)U64, DataFlowTag(495)U64)U64(506) BEGIN
 // (op 506) Assign dim(494) = Add(DataFlowTag(494)U64, DataFlowTag(495)U64)U64
 // (op 506) Generate Add(DataFlowTag(494)U64, DataFlowTag(495)U64)U64 into DataFlowTag494: VGPR Value: UInt64 x 1: v[18:19]
 // (op 506) reg expression
 // (op 506) Add({DataFlowTag494: v[18:19]:U64}, 64:U64)U64
 // (op 506) Add: dest (VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 506)      lhs (DataFlowTag494: VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 506)      rhs (Literal Value: UInt64 x 0: 64)
// Allocated : 2 SGPRs (Value: Bool64) (op 506): s14, s15
v_add_co_u32 v18, s[14:15], 64, v18 // (op 506) least significant half
v_addc_co_u32 v19, s[14:15], 0, v19, s[14:15] // (op 506) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 506): s14, s15

 // (op 226) Assign VGPR Add(DataFlowTag(494)U64, DataFlowTag(495)U64)U64(506) END
 // (op 226) end: generate({468})
 // (op 175) SetCoordinate(226) END
 // (op 175) NOP(470) BEGIN
 // (op 470) generate({})
 // (op 470) end: generate({})
 // (op 175) NOP(470) END
 // (op 175) Barrier(1213) BEGIN
s_barrier  // (op 1213) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 175) Barrier(1213) END
 // (op 175) SetCoordinate(231) BEGIN
 // (op 231) generate({})
 // (op 231) end: generate({})
 // (op 231) generate({474})
 // (op 231) LoadTileDirect2LDS Value: FP8x4(474) BEGIN
 // (op 474) GEN: LoadTileDirect2LDS
 // (op 474) GEN: loadMacroTileDirect2LDS OP 474 LDS 25 MacroTile 456
 // (op 474) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 474) Generate Convert(ShiftL(Tensor_2_stride_1_3:I64, 4:U32)I64)U64 into Stride518: VGPR Value: UInt64 x 1: (unallocated)
 // (op 474) reg expression
 // (op 474) ShiftL({Tensor_2_stride_1_3: s[8:9]:I64}, 4:U32)I64
 // (op 474) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 474)         value (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 474)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 474): s14, s15
s_lshl_b64 s[14:15], s[8:9], 4 // (op 474) 
// Allocated Stride518: 2 VGPRs (Value: UInt64) (op 474): v12, v13
v_mov_b64 v[12:13], s[14:15] // (op 474) convert
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 474): s14, s15

 // (op 474) Generate 1:U64 into nullptr
 // (op 474) Generate 1024:U32 into nullptr
 // (op 474) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer2LDS
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 1024
 // 	info.data = Literal Value: UInt32 x 0: 0
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride518: VGPR Value: UInt64 x 1: v[12:13]
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
 // 	info.bufDesc = DataFlowTag516: SGPR Buffer: None x 1: s[44:47]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 1
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 517: v**UNALLOCATED**
// Allocated Offset517: 2 VGPRs (Value: UInt64) (op 474): v20, v21
v_mov_b64 v[20:21], v[14:15] // (op 474) 
// Allocated : 2 VGPRs (Value: UInt64) (op 474): v26, v27
v_mov_b64 v[26:27], v[20:21] // (op 474) 
 // (op 474) Generate Add(0:U32, {DataFlowTag596Flatten: s1:U32})U32 into nullptr
 // (op 474)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride518: v[12:13]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
 // (op 474) Generate {DataFlowTag596Flatten: s1:U32} into M0 Value: UInt32 x 1: m0
s_mov_b32 m0, s1 // (op 474) reg expression
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 474) Load value direct to lds
// VMEM: Expected complete at 4677 (current 4197)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 474) Generate Add(v[26:27]:U64, v12:R)U64 into VGPR Value: UInt64 x 1: v[26:27]
 // (op 474) reg expression
 // (op 474) reg expression
 // (op 474) Add(v[26:27]:U64, v12:R)U64
 // (op 474) Add: dest (VGPR Value: UInt64 x 1: v[26:27]) = 
 // (op 474)      lhs (VGPR Value: UInt64 x 1: v[26:27]) 
 // (op 474)      rhs (VGPR Value: Raw32 x 1: v12)
// Allocated : 2 SGPRs (Value: Bool64) (op 474): s14, s15
v_add_co_u32 v26, s[14:15], v26, v12 // (op 474) least significant half
v_addc_co_u32 v27, s[14:15], 0, v27, s[14:15] // (op 474) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 474): s14, s15

 // (op 474) Generate Add(m0:U32, 1024:U32)U32 into M0 Value: UInt32 x 1: m0
 // (op 474) reg expression
 // (op 474) Add(m0:U32, 1024:U32)U32
 // (op 474) Expression temporary in special register (M0)
 // Lock M0
s_add_u32 m0, m0, 1024 // (op 474) 
 // (op 474) Expression temporary in special register
 // Unlock None
buffer_load_dwordx4 v26, s[44:47], 0 offen offset: 0 lds // (op 474) Load value direct to lds
// VMEM: Expected stall of 464. CBNW: 0, Inc: 3
// VMEM: Expected complete at 5145 (current 4665)
// Extra dsts: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // Freeing Stride518: 2 VGPRs (Value: UInt64) (op 474): v12, v13

 // Freeing : 2 VGPRs (Value: UInt64) (op 474): v26, v27

 // (op 231) LoadTileDirect2LDS Value: FP8x4(474) END
 // (op 231) Deallocate{}(1375) BEGIN
 // (op 1375) Deallocate 519
 // Deleting tag 519
 // (op 1375) Deallocate 598
 // Deleting tag 598
 // (op 1375) Deallocate 456
 // Deleting tag 456
 // (op 1375) Deallocate 517
 // Freeing Offset517: 2 VGPRs (Value: UInt64) (op 474): v20, v21

 // Deleting tag 517
 // (op 231) Deallocate{}(1375) END
 // (op 231) Assign VGPR Add(DataFlowTag(513)U64, DataFlowTag(514)U64)U64(575) BEGIN
 // (op 575) Assign dim(513) = Add(DataFlowTag(513)U64, DataFlowTag(514)U64)U64
 // (op 575) Generate Add(DataFlowTag(513)U64, DataFlowTag(514)U64)U64 into DataFlowTag513: VGPR Value: UInt64 x 1: v[14:15]
 // (op 575) reg expression
 // (op 575) Add({DataFlowTag513: v[14:15]:U64}, 64:U64)U64
 // (op 575) Add: dest (VGPR Value: UInt64 x 1: v[14:15]) = 
 // (op 575)      lhs (DataFlowTag513: VGPR Value: UInt64 x 1: v[14:15]) 
 // (op 575)      rhs (Literal Value: UInt64 x 0: 64)
// Allocated : 2 SGPRs (Value: Bool64) (op 575): s14, s15
v_add_co_u32 v14, s[14:15], 64, v14 // (op 575) least significant half
v_addc_co_u32 v15, s[14:15], 0, v15, s[14:15] // (op 575) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 575): s14, s15

 // (op 231) Assign VGPR Add(DataFlowTag(513)U64, DataFlowTag(514)U64)U64(575) END
 // (op 231) end: generate({474})
 // (op 175) SetCoordinate(231) END
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
 // (op 175) NOP(476) BEGIN
 // (op 476) generate({})
 // (op 476) end: generate({})
 // (op 175) NOP(476) END
// Allocated Offset536: 1 VGPR (Value: UInt32) (op 195): v16
v_mov_b32 v16, v25 // (op 195) 
 // (op 195) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset536: VGPR Value: UInt32 x 1: v16
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

// Allocated : 1 VGPR (Value: E8M0) (op 195): v24
buffer_load_ubyte v24, v16, s[48:51], 0 offen offset: 0 // (op 195) Load value
// VMEM: Expected complete at 5149 (current 4669)
 // (op 236) LoadTiled Value: E8M0(195) END
 // (op 236) Deallocate{}(1315) BEGIN
 // (op 1315) Deallocate 538
 // Deleting tag 538
 // (op 1315) Deallocate 536
 // Freeing Offset536: 1 VGPR (Value: UInt32) (op 195): v16

 // Deleting tag 536
 // (op 236) Deallocate{}(1315) END
 // (op 236) Assign VGPR Add(DataFlowTag(532)U32, Convert(DataFlowTag(533)U64)U32)U32(639) BEGIN
 // (op 639) Assign dim(532) = Add(DataFlowTag(532)U32, Convert(DataFlowTag(533)U64)U32)U32
 // (op 639) Generate Add(DataFlowTag(532)U32, Convert(DataFlowTag(533)U64)U32)U32 into DataFlowTag532: VGPR Value: UInt32 x 1: v25
 // (op 639) reg expression
 // (op 639) Add({DataFlowTag532: v25:U32}, 2:U32)U32
v_add_u32 v25, 2, v25 // (op 639) 
 // (op 236) Assign VGPR Add(DataFlowTag(532)U32, Convert(DataFlowTag(533)U64)U32)U32(639) END
 // (op 236) end: generate({195})
 // (op 175) SetCoordinate(236) END
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
 // (op 175) Barrier(1230) BEGIN
s_barrier  // (op 1230) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 175) Barrier(1230) END
// Allocated Offset555: 1 VGPR (Value: UInt32) (op 196): v16
v_mov_b32 v16, v17 // (op 196) 
 // (op 196) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset555: VGPR Value: UInt32 x 1: v16
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
 // 	info.data = VGPR Value: E8M0 x 1: v24
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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
 // 	info.data = VGPR Value: E8M0 x 1: v24
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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

// Allocated : 1 VGPR (Value: E8M0) (op 196): v12
buffer_load_ubyte v12, v16, s[52:55], 0 offen offset: 0 // (op 196) Load value
// VMEM: Expected stall of 5. CBNW: 2, Inc: 3
// VMEM: Expected complete at 5158 (current 4678)
 // (op 241) LoadTiled Value: E8M0(196) END
 // (op 241) Deallocate{}(1317) BEGIN
 // (op 1317) Deallocate 557
 // Deleting tag 557
 // (op 1317) Deallocate 555
 // Freeing Offset555: 1 VGPR (Value: UInt32) (op 196): v16

 // Deleting tag 555
 // (op 241) Deallocate{}(1317) END
 // (op 241) Assign VGPR Add(DataFlowTag(551)U32, Convert(DataFlowTag(552)U64)U32)U32(703) BEGIN
 // (op 703) Assign dim(551) = Add(DataFlowTag(551)U32, Convert(DataFlowTag(552)U64)U32)U32
 // (op 703) Generate Add(DataFlowTag(551)U32, Convert(DataFlowTag(552)U64)U32)U32 into DataFlowTag551: VGPR Value: UInt32 x 1: v17
 // (op 703) reg expression
 // (op 703) Add({DataFlowTag551: v17:U32}, 2:U32)U32
v_add_u32 v17, 2, v17 // (op 703) 
 // (op 241) Assign VGPR Add(DataFlowTag(551)U32, Convert(DataFlowTag(552)U64)U32)U32(703) END
 // (op 241) end: generate({196})
 // (op 175) SetCoordinate(241) END
s_waitcnt vmcnt(1)
ds_write_b8 v22, v24 offset:8256 // (op 192) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 4757 (current 4662)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 175) StoreLDSTile Value: E8M0(192) END
 // (op 175) Barrier(1203) BEGIN
 // (op 175) Deallocate{}(1311) BEGIN
 // (op 1311) Deallocate 614
 // Deleting tag 614
 // (op 1311) Deallocate 458
 // Freeing : 1 VGPR (Value: E8M0) (op 195): v24

 // Deleting tag 458
 // (op 175) Deallocate{}(1311) END
s_barrier  // (op 1203) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 175) Barrier(1203) END
 // (op 175) Barrier(1247) BEGIN
 // (op 175) SetCoordinate(182) BEGIN
 // (op 182) generate({})
 // (op 182) end: generate({})
 // (op 182) generate({183})
 // (op 182) LoadLDSTile{Value: BF8x4}(183) BEGIN
 // (op 183) GEN: loadMacroTileWAVELDS OP 183 LDS 17 WaveTile 105
 // (op 183) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 183) Generate 1:U32 into nullptr
 // tag 462: v**UNALLOCATED**
 // (op 183) 	Dir = Load
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

 // (op 183) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag588: VGPR Value: UInt32 x 1: v3
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
 // (op 183) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

s_barrier  // (op 1247) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 175) Barrier(1247) END
// Allocated : 8 VGPRs (Value: BF8x4) (op 183): v26, v27, v28, v29, v30, v31, v32, v33
ds_read_b128 v[26:29], v3 offset:2048 // (op 183) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4760 (current 4665)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
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

 // (op 193) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v12
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
 // (op 193) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_read_b128 v[30:33], v3 offset:2080 // (op 183) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4761 (current 4666)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 2048, Size: 2048)
 // (op 182) LoadLDSTile{Value: BF8x4}(183) END
 // (op 182) end: generate({183})
 // (op 175) SetCoordinate(182) END
s_waitcnt vmcnt(0)
ds_write_b8 v23, v12 offset:8192 // (op 193) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 91. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 4854 (current 4759)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 175) StoreLDSTile Value: E8M0(193) END
 // (op 175) Barrier(1220) BEGIN
 // (op 175) Deallocate{}(1313) BEGIN
 // (op 1313) Deallocate 630
 // Deleting tag 630
 // (op 1313) Deallocate 460
 // Freeing : 1 VGPR (Value: E8M0) (op 196): v12

 // Deleting tag 460
 // (op 175) Deallocate{}(1313) END
s_barrier  // (op 1220) 
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 175) Barrier(1220) END
 // (op 175) SetCoordinate(184) BEGIN
 // (op 184) generate({})
 // (op 184) end: generate({})
 // (op 184) generate({185})
 // (op 184) LoadLDSTile{Value: FP8x4}(185) BEGIN
 // (op 185) GEN: loadMacroTileWAVELDS OP 185 LDS 25 WaveTile 171
 // (op 185) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 185) Generate 1:U32 into nullptr
 // tag 464: v**UNALLOCATED**
 // (op 185) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: (unallocated)
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

 // (op 185) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag604: VGPR Value: UInt32 x 1: v4
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
 // (op 185) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

// Allocated : 8 VGPRs (Value: FP8x4) (op 185): v34, v35, v36, v37, v38, v39, v40, v41
ds_read_b128 v[34:37], v4 // (op 185) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4856 (current 4761)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
ds_read_b128 v[38:41], v4 offset:32 // (op 185) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4857 (current 4762)
// Extra srcs: LDS Value: FP8x4 x 0: LDS:(Offset: 0, Size: 2048)
 // (op 184) LoadLDSTile{Value: FP8x4}(185) END
 // (op 184) end: generate({185})
 // (op 175) SetCoordinate(184) END
 // (op 175) Barrier(1237) BEGIN
s_waitcnt lgkmcnt(5)
s_barrier  // (op 1237) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 175) Barrier(1237) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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
 // 	info.offset = Literal Value: UInt32 x 0: 8256
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

// Allocated : 1 VGPR (Value: E8M0) (op 187): v16
ds_read_u8 v16, v0 offset:8256 // (op 187) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected stall of 90. CBNW: 2, Inc: 3
// DSMEM: Expected complete at 4950 (current 4855)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8256, Size: 64)
 // (op 186) LoadLDSTile{Value: E8M0}(187) END
 // (op 186) end: generate({187})
 // (op 175) SetCoordinate(186) END
 // (op 175) Barrier(1254) BEGIN
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1254) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 175) Barrier(1254) END
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
 // (op 189) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: E8M0) (op 189): v24
ds_read_u8 v24, v6 offset:8192 // (op 189) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4953 (current 4858)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 8192, Size: 64)
 // (op 188) LoadLDSTile{Value: E8M0}(189) END
 // (op 188) end: generate({189})
 // (op 175) SetCoordinate(188) END
 // (op 175) Multiply(190) BEGIN
 // (op 190) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile4xFP8, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)S into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[26:33], v[34:41], a[0:15], v16, v24 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b000 // (op 190) 
 // (op 175) Multiply(190) END
 // (op 175) Deallocate{}(1309) BEGIN
 // (op 1309) Deallocate 468
 // Freeing : 1 VGPR (Value: E8M0) (op 189): v24

 // Deleting tag 468
 // (op 1309) Deallocate 466
 // Freeing : 1 VGPR (Value: E8M0) (op 187): v16

 // Deleting tag 466
 // (op 1309) Deallocate 464
 // Freeing : 8 VGPRs (Value: FP8x4) (op 185): v34, v35, v36, v37, v38, v39, v40, v41

 // Deleting tag 464
 // (op 1309) Deallocate 462
 // Freeing : 8 VGPRs (Value: BF8x4) (op 183): v26, v27, v28, v29, v30, v31, v32, v33

 // Deleting tag 462
 // (op 175) Deallocate{}(1309) END
 // (op 175) end: generate({177, 1197})
 // (op 175) For Loop Increment
 // (op 175) generate({})
 // (op 175) end: generate({})
 // (op 175) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175 if true)
 // (op 175) Generate LessThan(DataFlowTag(452)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 175) LessThan({DataFlowTag452: s18:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32)BL
 // (op 175) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 175): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 175) 
// Allocated : 1 SGPR (Value: UInt32) (op 175): s19
s_mov_b32 s19, s14 // (op 175) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 175): s14, s15

s_cmp_lt_u32 s18, s19 // (op 175) 
 // Freeing : 1 SGPR (Value: UInt32) (op 175): s19

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 175) 
s_cbranch_scc1 RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175 // (op 175) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_175 if true)
RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_175:
 // (op 175) 
 // (op 175) Unlock For Loop
 // Unlock None
 // (op 511) ForLoopOp KLoopTail: LessThan(DataFlowTag(452)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL(175) END
 // (op 511) Deallocate{}(1307) BEGIN
 // (op 1307) Deallocate 452
 // Freeing DataFlowTag452: 1 SGPR (Value: UInt32) (op 176): s18

 // Deleting tag 452
 // (op 1307) Deallocate 631
 // Deleting tag 631
 // (op 1307) Deallocate 629
 // Deleting tag 629
 // (op 1307) Deallocate 628
 // Freeing DataFlowTag628: 1 VGPR (Value: UInt32) (op 961): v23

 // Deleting tag 628
 // (op 1307) Deallocate 615
 // Deleting tag 615
 // (op 1307) Deallocate 613
 // Deleting tag 613
 // (op 1307) Deallocate 612
 // Freeing DataFlowTag612: 1 VGPR (Value: UInt32) (op 908): v22

 // Deleting tag 612
 // (op 1307) Deallocate 599
 // Deleting tag 599
 // (op 1307) Deallocate 597
 // Deleting tag 597
 // (op 1307) Deallocate 596
 // Freeing DataFlowTag596: 1 SGPR (Value: UInt32) (op 856): s1

 // Deleting tag 596
 // (op 1307) Deallocate 583
 // Deleting tag 583
 // (op 1307) Deallocate 581
 // Deleting tag 581
 // (op 1307) Deallocate 580
 // Freeing DataFlowTag580: 1 SGPR (Value: UInt32) (op 804): s0

 // Deleting tag 580
 // (op 1307) Deallocate 556
 // Deleting tag 556
 // (op 1307) Deallocate 552
 // Deleting tag 552
 // (op 1307) Deallocate 551
 // Freeing DataFlowTag551: 1 VGPR (Value: UInt32) (op 711): v17

 // Deleting tag 551
 // (op 1307) Deallocate 537
 // Deleting tag 537
 // (op 1307) Deallocate 558
 // Deleting tag 558
 // (op 1307) Deallocate 539
 // Deleting tag 539
 // (op 1307) Deallocate 518
 // Deleting tag 518
 // (op 1307) Deallocate 494
 // Freeing DataFlowTag494: 2 VGPRs (Value: UInt64) (op 516): v18, v19

 // Deleting tag 494
 // (op 1307) Deallocate 495
 // Deleting tag 495
 // (op 1307) Deallocate 499
 // Deleting tag 499
 // (op 1307) Deallocate 501
 // Deleting tag 501
 // (op 1307) Deallocate 513
 // Freeing DataFlowTag513: 2 VGPRs (Value: UInt64) (op 583): v14, v15

 // Deleting tag 513
 // (op 1307) Deallocate 514
 // Deleting tag 514
 // (op 1307) Deallocate 520
 // Deleting tag 520
 // (op 1307) Deallocate 532
 // Freeing DataFlowTag532: 1 VGPR (Value: UInt32) (op 647): v25

 // Deleting tag 532
 // (op 1307) Deallocate 533
 // Deleting tag 533
 // (op 511) Deallocate{}(1307) END
 // (op 511) end: generate({505})
 // (op 511) Unlock Scope 511
 // Unlock None
 // (op 222) Scope(511) END
 // (op 222) end: generate({511})
 // (op 221) SetCoordinate(222) END
 // (op 221) end: generate({222})
 // (op 344) SetCoordinate(221) END
 // (op 344) Deallocate{}(1319) BEGIN
 // (op 1319) Deallocate 33
 // Deleting tag 33
 // (op 1319) Deallocate 41
 // Deleting tag 41
 // (op 1319) Deallocate 25
 // Deleting tag 25
 // (op 1319) Deallocate 144
 // Deleting tag 144
 // (op 1319) Deallocate 622
 // Deleting tag 622
 // (op 1319) Deallocate 590
 // Deleting tag 590
 // (op 1319) Deallocate 535
 // Freeing DataFlowTag535: 4 SGPRs (Buffer: None) (op 672): s48, s49, s50, s51

 // Deleting tag 535
 // (op 1319) Deallocate 433
 // Deleting tag 433
 // (op 1319) Deallocate 554
 // Freeing DataFlowTag554: 4 SGPRs (Buffer: None) (op 736): s52, s53, s54, s55

 // Deleting tag 554
 // (op 1319) Deallocate 4
 // Deleting tag 4
 // (op 1319) Deallocate 516
 // Freeing DataFlowTag516: 4 SGPRs (Buffer: None) (op 608): s44, s45, s46, s47

 // Deleting tag 516
 // (op 1319) Deallocate 270
 // Deleting tag 270
 // (op 1319) Deallocate 437
 // Deleting tag 437
 // (op 1319) Deallocate 497
 // Freeing DataFlowTag497: 4 SGPRs (Buffer: None) (op 544): s40, s41, s42, s43

 // Deleting tag 497
 // (op 1319) Deallocate 78
 // Deleting tag 78
 // (op 1319) Deallocate 431
 // Deleting tag 431
 // (op 1319) Deallocate 17
 // Deleting tag 17
 // (op 1319) Deallocate 159
 // Deleting tag 159
 // (op 1319) Deallocate 210
 // Deleting tag 210
 // (op 1319) Deallocate 435
 // Deleting tag 435
 // (op 1319) Deallocate 3
 // Deleting tag 3
 // (op 1319) Deallocate 638
 // Deleting tag 638
 // (op 1319) Deallocate 93
 // Deleting tag 93
 // (op 1319) Deallocate 285
 // Deleting tag 285
 // (op 1319) Deallocate 225
 // Deleting tag 225
 // (op 1319) Deallocate 606
 // Deleting tag 606
 // (op 344) Deallocate{}(1319) END
 // (op 344) Deallocate{Tensor_4_stride_0_6, Tensor_0_stride_0_0, Tensor_7_pointer, Tensor_0_pointer, Tensor_2_pointer, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_0_size_0_1, Tensor_2_stride_1_3, Tensor_4_pointer}(1451) BEGIN
 // (op 1451) Deallocate Tensor_4_stride_0_6
 // Freeing Tensor_4_stride_0_6: 2 SGPRs (Value: Raw32): s20, s21

 // (op 1451) Deallocate Tensor_0_stride_0_0
 // Freeing Tensor_0_stride_0_0: 2 SGPRs (Value: Raw32): s2, s3

 // (op 1451) Deallocate Tensor_7_pointer
 // Freeing Tensor_7_pointer: 2 SGPRs (Value: Raw32): s26, s27

 // (op 1451) Deallocate Tensor_0_pointer
 // Freeing Tensor_0_pointer: 2 SGPRs (Value: Raw32): s4, s5

 // (op 1451) Deallocate Tensor_2_pointer
 // Freeing Tensor_2_pointer: 2 SGPRs (Value: Raw32): s10, s11

 // (op 1451) Deallocate Tensor_7_stride_1_8
 // Freeing Tensor_7_stride_1_8: 2 SGPRs (Value: Raw32): s24, s25

 // (op 1451) Deallocate Tensor_0_size_1_14
 // Freeing Tensor_0_size_1_14: 2 SGPRs (Value: Raw32): s28, s29

 // (op 1451) Deallocate Tensor_0_size_0_1
 // Freeing Tensor_0_size_0_1: 2 SGPRs (Value: Raw32): s6, s7

 // (op 1451) Deallocate Tensor_2_stride_1_3
 // Freeing Tensor_2_stride_1_3: 2 SGPRs (Value: Raw32): s8, s9

 // (op 1451) Deallocate Tensor_4_pointer
 // Freeing Tensor_4_pointer: 2 SGPRs (Value: Raw32): s22, s23

 // (op 344) Deallocate{Tensor_4_stride_0_6, Tensor_0_stride_0_0, Tensor_7_pointer, Tensor_0_pointer, Tensor_2_pointer, Tensor_7_stride_1_8, Tensor_0_size_1_14, Tensor_0_size_0_1, Tensor_2_stride_1_3, Tensor_4_pointer}(1451) END
 // (op 344) Barrier(1321) BEGIN
s_barrier  // (op 1321) 
 // (op 344) Barrier(1321) END
 // (op 344) end: generate({349})
 // (op 344) Unlock Scope 344
 // Unlock None
 // (op 1) Scope(344) END
 // (op 1) Deallocate{}(1353) BEGIN
 // (op 1353) Deallocate 681
 // Deleting tag 681
 // (op 1353) Deallocate 639
 // Deleting tag 639
 // (op 1353) Deallocate 620
 // Freeing DataFlowTag620: 1 VGPR (Value: UInt32) (op 936): v0

 // Deleting tag 620
 // (op 1353) Deallocate 661
 // Deleting tag 661
 // (op 1353) Deallocate 679
 // Deleting tag 679
 // (op 1353) Deallocate 659
 // Deleting tag 659
 // (op 1353) Deallocate 607
 // Deleting tag 607
 // (op 1353) Deallocate 623
 // Deleting tag 623
 // (op 1353) Deallocate 621
 // Deleting tag 621
 // (op 1353) Deallocate 588
 // Freeing DataFlowTag588: 1 VGPR (Value: UInt32) (op 831): v3

 // Deleting tag 588
 // (op 1353) Deallocate 605
 // Deleting tag 605
 // (op 1353) Deallocate 687
 // Deleting tag 687
 // (op 1353) Deallocate 589
 // Deleting tag 589
 // (op 1353) Deallocate 670
 // Freeing DataFlowTag670: 1 VGPR (Value: UInt32) (op 1101): v8

 // Deleting tag 670
 // (op 1353) Deallocate 658
 // Freeing DataFlowTag658: 1 VGPR (Value: UInt32) (op 1062): v7

 // Deleting tag 658
 // (op 1353) Deallocate 637
 // Deleting tag 637
 // (op 1353) Deallocate 636
 // Freeing DataFlowTag636: 1 VGPR (Value: UInt32) (op 989): v6

 // Deleting tag 636
 // (op 1353) Deallocate 686
 // Freeing DataFlowTag686: 1 VGPR (Value: UInt32) (op 1149): v10

 // Deleting tag 686
 // (op 1353) Deallocate 673
 // Deleting tag 673
 // (op 1353) Deallocate 678
 // Freeing DataFlowTag678: 1 VGPR (Value: UInt32) (op 1125): v5

 // Deleting tag 678
 // (op 1353) Deallocate 591
 // Deleting tag 591
 // (op 1353) Deallocate 689
 // Deleting tag 689
 // (op 1353) Deallocate 671
 // Deleting tag 671
 // (op 1353) Deallocate 604
 // Freeing DataFlowTag604: 1 VGPR (Value: UInt32) (op 883): v4

 // Deleting tag 604
 // (op 1) Deallocate{}(1353) END
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
s_cbranch_vccz RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_135 // (op 135) Condition: False, jump to RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_135
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
 // (op 352) Deallocate{}(1287) BEGIN
 // (op 1287) Deallocate 59
 // Freeing DataFlowTag59: 16 VGPRs (Value: Float) (op 50): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // Deleting tag 59
 // (op 352) Deallocate{}(1287) END
 // (op 352) NOP(999) BEGIN
 // (op 999) generate({})
 // (op 999) end: generate({})
 // (op 352) NOP(999) END
 // (op 352) NOP(996) BEGIN
 // (op 996) generate({})
 // (op 996) end: generate({})
 // (op 352) NOP(996) END
 // (op 352) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(1003) BEGIN
 // (op 1003) Assign dim(640) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}
 // tag 640: v**UNALLOCATED**
 // (op 1003) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32} into DataFlowTag640: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1003) reg expression
 // (op 1003) reg expression
 // (op 1003) reg expression
 // (op 1003) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1003) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 1003): v0
v_lshrrev_b32 v0, 6, v1 // (op 1003) 
 // (op 1003) convert
 // (op 1003) BEGIN: Tile[1]
 // (op 1003) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1003): v10
v_and_b32 v10, 63, v1 // (op 1003) 
 // (op 1003) END: Tile[1]
 // (op 1003) LogicalShiftR({Tile[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1003): v3
v_lshrrev_b32 v3, 5, v10 // (op 1003) 
 // (op 1003) BEGIN: Flatten
 // (op 1003) {Flatten: ShiftL(v3:U32, 2:U32)U32}
 // (op 1003) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1003)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 1003)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 1003): v4
v_lshlrev_b32 v4, 2, v3 // (op 1003) 
 // (op 1003) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1003): v3

 // (op 1003) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1003): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 1003) 
 // (op 1003) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1003): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 1003): v0

 // (op 1003) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1003): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 1003) 
 // (op 1003) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1003): v3

 // (op 1003) LogicalShiftR({Tile[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1003): v3
v_lshrrev_b32 v3, 2, v10 // (op 1003) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1003): v10

 // (op 1003) BEGIN: Tile[1]
 // (op 1003) {Tile[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1003): v10
v_and_b32 v10, 7, v3 // (op 1003) 
 // (op 1003) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1003): v3

 // (op 1003) BEGIN: Tile[1]
 // (op 1003) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1003): v3
v_and_b32 v3, 3, v1 // (op 1003) 
 // (op 1003) END: Tile[1]
 // (op 1003) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 1003): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 1003) 
 // (op 1003) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1003): v3

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1003): v10

 // (op 1003) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1003): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 1003) 
 // (op 1003) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 1003): v4

 // (op 1003) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 1003): v10
v_mul_lo_u32 v10, v3, s30 // (op 1003) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1003): v3

 // (op 1003) BEGIN: Join
// Allocated DataFlowTag640: 1 VGPR (Value: UInt32) (op 1003): v3
v_add_lshl_u32 v3, v0, v10, 1 // (op 1003) 
 // (op 1003) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 1003): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1003): v0

 // (op 352) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(1003) END
 // (op 352) Assign VGPR 16:U64(1004) BEGIN
 // (op 1004) Assign dim(641) = 16:U64
 // (op 1004) Assign stride expression
 // (op 352) Assign VGPR 16:U64(1004) END
 // (op 352) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(1005) BEGIN
 // (op 1005) Assign dim(643) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 643: s**UNALLOCATED**
 // (op 1005) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag643: SGPR Buffer: None x 1: (unallocated)
 // (op 1005) reg expression
 // (op 1005) reg expression
 // (op 1005) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 1005) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 1005)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 1005)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 1005): s14, s15
s_mul_i32 s15, s30, s13 // (op 1005) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 1005): s0
s_mul_i32 s0, s31, s12 // (op 1005) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1005) least significant
// Allocated : 1 SGPR (Value: Int32) (op 1005): s1
s_mul_hi_u32 s1, s30, s12 // (op 1005) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1005) most significant: sum
s_add_u32 s15, s15, s1 // (op 1005) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 1005): s1

 // Freeing : 1 SGPR (Value: Int32) (op 1005): s0

 // (op 1005) ShiftL(s[14:15]:I64, 1:U32)I64
 // (op 1005) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 1005)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 1005)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 2 SGPRs (Value: Int64) (op 1005): s0, s1
s_lshl_b64 s[0:1], s[14:15], 1 // (op 1005) 
 // Freeing : 2 SGPRs (Value: Int64) (op 1005): s14, s15

 // (op 1005) reg expression
 // (op 1005) reg expression
// Allocated DataFlowTag643: 4 SGPRs (Buffer: None) (op 1005): s8, s9, s10, s11
s_mov_b32 s8, s32 // (op 1005) reg expression
s_mov_b32 s9, s33 // (op 1005) reg expression
s_mov_b32 s10, s0 // (op 1005) reg expression
s_mov_b32 s11, 131072 // (op 1005) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 1005): s0, s1

 // (op 352) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(1005) END
 // (op 352) NOP(997) BEGIN
 // (op 997) generate({})
 // (op 997) end: generate({})
 // (op 352) NOP(997) END
 // (op 352) Assign VGPR 2:U64(1010) BEGIN
 // (op 1010) Assign dim(645) = 2:U64
 // (op 1010) Assign stride expression
 // (op 352) Assign VGPR 2:U64(1010) END
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
// VMEM: Expected complete at 5928 (current 5448)
buffer_store_dwordx2 v[30:31], v3, s[8:11], 0 offen offset:16 // (op 58) Store value
// VMEM: Expected complete at 5929 (current 5449)
buffer_store_dwordx2 v[32:33], v3, s[8:11], 0 offen offset:32 // (op 58) Store value
// VMEM: Expected complete at 5930 (current 5450)
buffer_store_dwordx2 v[34:35], v3, s[8:11], 0 offen offset:48 // (op 58) Store value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 6409 (current 5929)
 // Freeing DataFlowTag59: 8 VGPRs (Value: Halfx2) (op 58): v28, v29, v30, v31, v32, v33, v34, v35

 // (op 352) StoreTiled Value: Halfx2(58) END
 // (op 352) Deallocate{}(1289) BEGIN
 // (op 1289) Deallocate 644
 // Deleting tag 644
 // (op 1289) Deallocate 645
 // Deleting tag 645
 // (op 1289) Deallocate 641
 // Deleting tag 641
 // (op 1289) Deallocate 640
 // Freeing DataFlowTag640: 1 VGPR (Value: UInt32) (op 1003): v3

 // Deleting tag 640
 // (op 1289) Deallocate 61
 // Freeing DataFlowTag59: 16 VGPRs (Value: Half) (op 56): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // Deleting tag 61
 // (op 352) Deallocate{}(1289) END
 // (op 352) end: generate({355})
 // (op 352) Unlock Scope 352
 // Unlock None
 // (op 135) Scope(352) END
 // (op 135) end: generate({352})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 135) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 135) 
s_branch RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_135 // (op 135) Condition: Done, jump to RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_135
RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_135:
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
 // (op 358) Scope(769) BEGIN
 // (op 769) Lock Scope 769
 // Lock Branch
 // (op 769) generate({766})
 // (op 769) NOP(766) BEGIN
 // (op 766) generate({})
 // (op 766) end: generate({})
 // (op 769) NOP(766) END
 // (op 769) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(326)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32}(774) BEGIN
 // (op 774) Assign dim(570) = {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(326)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32}
 // tag 570: v**UNALLOCATED**
 // (op 774) Generate {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(326)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32} into DataFlowTag570: VGPR Value: UInt32 x 1: (unallocated)
 // (op 774) reg expression
 // (op 774) reg expression
 // (op 774) reg expression
 // (op 774) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 774) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 774): v0
v_lshrrev_b32 v0, 6, v1 // (op 774) 
 // (op 774) convert
 // (op 774) BEGIN: Flatten[1]
 // (op 774) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 774): v10
v_and_b32 v10, 63, v1 // (op 774) 
 // (op 774) END: Flatten[1]
 // (op 774) LogicalShiftR({Flatten[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 774): v3
v_lshrrev_b32 v3, 5, v10 // (op 774) 
 // (op 774) BEGIN: Tile
 // (op 774) {Tile: ShiftL(v3:U32, 2:U32)U32}
 // (op 774) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 774)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 774)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 774): v4
v_lshlrev_b32 v4, 2, v3 // (op 774) 
 // (op 774) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 774): v3

 // (op 774) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 774): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 774) 
 // (op 774) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 774): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 774): v0

 // (op 774) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 774): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 774) 
 // (op 774) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 774): v3

 // (op 774) LogicalShiftR({Flatten[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 774): v3
v_lshrrev_b32 v3, 2, v10 // (op 774) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 774): v10

 // (op 774) BEGIN: Flatten[1]
 // (op 774) {Flatten[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 774): v10
v_and_b32 v10, 7, v3 // (op 774) 
 // (op 774) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 774): v3

 // (op 774) BEGIN: Flatten[1]
 // (op 774) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 774): v3
v_and_b32 v3, 3, v1 // (op 774) 
 // (op 774) END: Flatten[1]
 // (op 774) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 774): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 774) 
 // (op 774) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 774): v3

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 774): v10

 // (op 774) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 774): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 774) 
 // (op 774) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 774): v4

 // (op 774) Multiply({Tile: v3:U32}, { convertForwardSubsetTensor_10_stride_1_10: s34:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 774): v10
v_mul_lo_u32 v10, v3, s34 // (op 774) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 774): v3

 // (op 774) BEGIN: Split
// Allocated DataFlowTag570: 1 VGPR (Value: UInt32) (op 774): v3
v_add_lshl_u32 v3, v0, v10, 1 // (op 774) 
 // (op 774) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 774): v10

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 774): v0

 // (op 769) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(326)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(326)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 1:U32)U32}(774) END
 // (op 769) Deallocate{}(1389) BEGIN
 // (op 1389) Deallocate 441
 // Deleting tag 441
 // (op 1389) Deallocate 439
 // Deleting tag 439
 // (op 1389) Deallocate 326
 // Deleting tag 326
 // (op 769) Deallocate{}(1389) END
 // (op 769) Assign VGPR 16:U64(775) BEGIN
 // (op 775) Assign dim(571) = 16:U64
 // (op 775) Assign stride expression
 // (op 769) Assign VGPR 16:U64(775) END
 // (op 769) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(776) BEGIN
 // (op 776) Assign dim(573) = Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 573: s**UNALLOCATED**
 // (op 776) Generate Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag573: SGPR Buffer: None x 1: (unallocated)
 // (op 776) reg expression
 // (op 776) reg expression
 // (op 776) Multiply({Tensor_10_stride_1_10: s[34:35]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 776) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 776)           lhs (Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]) 
 // (op 776)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 776): s14, s15
s_mul_i32 s15, s34, s13 // (op 776) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 776): s0
s_mul_i32 s0, s35, s12 // (op 776) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 776) least significant
// Allocated : 1 SGPR (Value: Int32) (op 776): s1
s_mul_hi_u32 s1, s34, s12 // (op 776) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 776) most significant: sum
s_add_u32 s15, s15, s1 // (op 776) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 776): s1

 // Freeing : 1 SGPR (Value: Int32) (op 776): s0

 // (op 776) ShiftL(s[14:15]:I64, 1:U32)I64
 // (op 776) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 776)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 776)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 2 SGPRs (Value: Int64) (op 776): s0, s1
s_lshl_b64 s[0:1], s[14:15], 1 // (op 776) 
 // Freeing : 2 SGPRs (Value: Int64) (op 776): s14, s15

 // (op 776) reg expression
 // (op 776) reg expression
// Allocated DataFlowTag573: 4 SGPRs (Buffer: None) (op 776): s4, s5, s6, s7
s_mov_b32 s4, s36 // (op 776) reg expression
s_mov_b32 s5, s37 // (op 776) reg expression
s_mov_b32 s6, s0 // (op 776) reg expression
s_mov_b32 s7, 131072 // (op 776) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 776): s0, s1

 // (op 769) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(776) END
 // (op 769) Deallocate{Tensor_10_pointer}(1455) BEGIN
 // (op 1455) Deallocate Tensor_10_pointer
 // Freeing Tensor_10_pointer: 2 SGPRs (Value: Raw32): s36, s37

 // (op 769) Deallocate{Tensor_10_pointer}(1455) END
 // (op 769) NOP(767) BEGIN
 // (op 767) generate({})
 // (op 767) end: generate({})
 // (op 769) NOP(767) END
 // (op 769) Assign VGPR 2:U64(781) BEGIN
 // (op 781) Assign dim(575) = 2:U64
 // (op 781) Assign stride expression
 // (op 769) Assign VGPR 2:U64(781) END
 // (op 769) LoadTiled Value: Halfx2(148) BEGIN
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
// VMEM: Expected complete at 6917 (current 6437)
buffer_load_dwordx2 v[14:15], v3, s[4:7], 0 offen offset:16 // (op 148) Load value
// VMEM: Expected complete at 6918 (current 6438)
buffer_load_dwordx2 v[16:17], v3, s[4:7], 0 offen offset:32 // (op 148) Load value
// VMEM: Expected complete at 6919 (current 6439)
buffer_load_dwordx2 v[18:19], v3, s[4:7], 0 offen offset:48 // (op 148) Load value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 7398 (current 6918)
 // (op 769) LoadTiled Value: Halfx2(148) END
 // (op 769) Deallocate{Tensor_10_stride_1_10}(1303) BEGIN
 // (op 1303) Deallocate 574
 // Deleting tag 574
 // (op 1303) Deallocate 571
 // Deleting tag 571
 // (op 1303) Deallocate 570
 // Freeing DataFlowTag570: 1 VGPR (Value: UInt32) (op 774): v3

 // Deleting tag 570
 // (op 1303) Deallocate 575
 // Deleting tag 575
 // (op 1303) Deallocate 573
 // Freeing DataFlowTag573: 4 SGPRs (Buffer: None) (op 776): s4, s5, s6, s7

 // Deleting tag 573
 // (op 1303) Deallocate 5
 // Deleting tag 5
 // (op 1303) Deallocate Tensor_10_stride_1_10
 // Freeing Tensor_10_stride_1_10: 2 SGPRs (Value: Raw32): s34, s35

 // (op 769) Deallocate{Tensor_10_stride_1_10}(1303) END
 // (op 769) end: generate({766})
 // (op 769) Unlock Scope 769
 // Unlock None
 // (op 358) Scope(769) END
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
 // (op 358) Deallocate{}(1305) BEGIN
 // (op 1305) Deallocate 421
 // Freeing : 8 VGPRs (Value: Halfx2) (op 148): v12, v13, v14, v15, v16, v17, v18, v19

 // Deleting tag 421
 // (op 358) Deallocate{}(1305) END
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
 // (op 358) Deallocate{}(1297) BEGIN
 // (op 1297) Deallocate 423
 // Freeing DataFlowTag423: 16 VGPRs (Value: Float) (op 149): v52, v53, v54, v55, v56, v57, v58, v59, v60, v61, v62, v63, v64, v65, v66, v67

 // Deleting tag 423
 // (op 1297) Deallocate 425
 // Freeing DataFlowTag425: 16 VGPRs (Value: Float) (op 144): v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51

 // Deleting tag 425
 // (op 358) Deallocate{}(1297) END
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
 // (op 358) Deallocate{}(1299) BEGIN
 // (op 1299) Deallocate 427
 // Freeing DataFlowTag427: 16 VGPRs (Value: Float) (op 145): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // Deleting tag 427
 // (op 358) Deallocate{}(1299) END
 // (op 358) NOP(1016) BEGIN
 // (op 1016) generate({})
 // (op 1016) end: generate({})
 // (op 358) NOP(1016) END
 // (op 358) NOP(1013) BEGIN
 // (op 1013) generate({})
 // (op 1013) end: generate({})
 // (op 358) NOP(1013) END
 // (op 358) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(1020) BEGIN
 // (op 1020) Assign dim(646) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}
 // tag 646: v**UNALLOCATED**
 // (op 1020) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32} into DataFlowTag646: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1020) reg expression
 // (op 1020) reg expression
 // (op 1020) reg expression
 // (op 1020) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1020) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 1020): v0
v_lshrrev_b32 v0, 6, v1 // (op 1020) 
 // (op 1020) convert
 // (op 1020) BEGIN: Tile[1]
 // (op 1020) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1020): v10
v_and_b32 v10, 63, v1 // (op 1020) 
 // (op 1020) END: Tile[1]
 // (op 1020) LogicalShiftR({Tile[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1020): v3
v_lshrrev_b32 v3, 5, v10 // (op 1020) 
 // (op 1020) BEGIN: Flatten
 // (op 1020) {Flatten: ShiftL(v3:U32, 2:U32)U32}
 // (op 1020) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1020)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 1020)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 1020): v4
v_lshlrev_b32 v4, 2, v3 // (op 1020) 
 // (op 1020) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1020): v3

 // (op 1020) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1020): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 1020) 
 // (op 1020) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1020): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 1020): v0

 // (op 1020) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1020): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 1020) 
 // (op 1020) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1020): v3

 // (op 1020) LogicalShiftR({Tile[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1020): v3
v_lshrrev_b32 v3, 2, v10 // (op 1020) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1020): v10

 // (op 1020) BEGIN: Tile[1]
 // (op 1020) {Tile[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1020): v10
v_and_b32 v10, 7, v3 // (op 1020) 
 // (op 1020) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1020): v3

 // (op 1020) BEGIN: Tile[1]
 // (op 1020) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1020): v3
v_and_b32 v3, 3, v1 // (op 1020) 
 // (op 1020) END: Tile[1]
 // (op 1020) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 1020): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 1020) 
 // (op 1020) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1020): v3

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 1020): v10

 // (op 1020) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 1020): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 1020) 
 // (op 1020) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 1020): v4

 // (op 1020) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 1020): v10
v_mul_lo_u32 v10, v3, s30 // (op 1020) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1020): v3

 // (op 1020) BEGIN: Join
// Allocated DataFlowTag646: 1 VGPR (Value: UInt32) (op 1020): v3
v_add_lshl_u32 v3, v0, v10, 1 // (op 1020) 
 // (op 1020) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 1020): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 1020): v0

 // (op 358) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(443)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(366)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(445)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(366)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 1:U32)U32}(1020) END
 // (op 358) Assign VGPR 16:U64(1021) BEGIN
 // (op 1021) Assign dim(647) = 16:U64
 // (op 1021) Assign stride expression
 // (op 358) Assign VGPR 16:U64(1021) END
 // (op 358) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(1022) BEGIN
 // (op 1022) Assign dim(643) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 1022) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag643: SGPR Buffer: None x 1: s[8:11]
 // (op 1022) reg expression
 // (op 1022) reg expression
 // (op 1022) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 1022) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 1022)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 1022)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 1022): s14, s15
s_mul_i32 s15, s30, s13 // (op 1022) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 1022): s0
s_mul_i32 s0, s31, s12 // (op 1022) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 1022) least significant
// Allocated : 1 SGPR (Value: Int32) (op 1022): s1
s_mul_hi_u32 s1, s30, s12 // (op 1022) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 1022) most significant: sum
s_add_u32 s15, s15, s1 // (op 1022) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 1022): s1

 // Freeing : 1 SGPR (Value: Int32) (op 1022): s0

 // (op 1022) ShiftL(s[14:15]:I64, 1:U32)I64
 // (op 1022) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 1022)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 1022)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 2 SGPRs (Value: Int64) (op 1022): s0, s1
s_lshl_b64 s[0:1], s[14:15], 1 // (op 1022) 
 // Freeing : 2 SGPRs (Value: Int64) (op 1022): s14, s15

 // (op 1022) reg expression
 // (op 1022) reg expression
s_mov_b32 s8, s32 // (op 1022) reg expression
s_mov_b32 s9, s33 // (op 1022) reg expression
s_mov_b32 s10, s0 // (op 1022) reg expression
s_mov_b32 s11, 131072 // (op 1022) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 1022): s0, s1

 // (op 358) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 1:U32)I64, width:32, offset:0)R, 131072:R)PB(1022) END
 // (op 358) NOP(1014) BEGIN
 // (op 1014) generate({})
 // (op 1014) end: generate({})
 // (op 358) NOP(1014) END
 // (op 358) Assign VGPR 2:U64(1027) BEGIN
 // (op 1027) Assign dim(649) = 2:U64
 // (op 1027) Assign stride expression
 // (op 358) Assign VGPR 2:U64(1027) END
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
// VMEM: Expected complete at 7954 (current 7474)
buffer_store_dwordx2 v[14:15], v3, s[8:11], 0 offen offset:16 // (op 147) Store value
// VMEM: Expected complete at 7955 (current 7475)
buffer_store_dwordx2 v[16:17], v3, s[8:11], 0 offen offset:32 // (op 147) Store value
// VMEM: Expected complete at 7956 (current 7476)
buffer_store_dwordx2 v[18:19], v3, s[8:11], 0 offen offset:48 // (op 147) Store value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 8435 (current 7955)
 // Freeing DataFlowTag427: 8 VGPRs (Value: Halfx2) (op 147): v12, v13, v14, v15, v16, v17, v18, v19

 // (op 358) StoreTiled Value: Halfx2(147) END
 // (op 358) Deallocate{}(1301) BEGIN
 // (op 1301) Deallocate 649
 // Deleting tag 649
 // (op 1301) Deallocate 648
 // Deleting tag 648
 // (op 1301) Deallocate 429
 // Freeing DataFlowTag427: 16 VGPRs (Value: Half) (op 146): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // Deleting tag 429
 // (op 1301) Deallocate 646
 // Freeing DataFlowTag646: 1 VGPR (Value: UInt32) (op 1020): v3

 // Deleting tag 646
 // (op 1301) Deallocate 647
 // Deleting tag 647
 // (op 358) Deallocate{}(1301) END
 // (op 358) end: generate({361})
 // (op 358) Unlock Scope 358
 // Unlock None
 // (op 135) Scope(358) END
 // (op 135) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1355) BEGIN
 // (op 1355) Deallocate 643
 // Freeing DataFlowTag643: 4 SGPRs (Buffer: None) (op 1005): s8, s9, s10, s11

 // Deleting tag 643
 // (op 1355) Deallocate 445
 // Deleting tag 445
 // (op 1355) Deallocate 443
 // Deleting tag 443
 // (op 1355) Deallocate 366
 // Deleting tag 366
 // (op 1355) Deallocate Tensor_21_pointer
 // Freeing Tensor_21_pointer: 2 SGPRs (Value: Raw32): s32, s33

 // (op 1355) Deallocate Tensor_21_stride_1_16
 // Freeing Tensor_21_stride_1_16: 2 SGPRs (Value: Raw32): s30, s31

 // (op 135) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1355) END
 // (op 135) end: generate({358})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 135) 
RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_135:
 // (op 135) 
 // (op 135) Unlock Conditional
 // Unlock None
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(44)S)BL(135) END
 // (op 1) Deallocate{Tensor_2_size_1_4}(1295) BEGIN
 // (op 1295) Deallocate 55
 // Freeing DataFlowTag55: 16 ACCVGPRs (Value: Float) (op 65): a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15

 // Deleting tag 55
 // (op 1295) Deallocate 53
 // Freeing : 1 VGPR (Value: Float) (op 43): v11

 // Deleting tag 53
 // (op 1295) Deallocate 44
 // Freeing : 1 VGPR (Value: Float) (op 2): v9

 // Deleting tag 44
 // (op 1295) Deallocate Tensor_2_size_1_4
 // Freeing Tensor_2_size_1_4: 2 SGPRs (Value: Raw32): s12, s13

 // (op 1) Deallocate{Tensor_2_size_1_4}(1295) END
 // (op 1) end: generate({827, 879, 932, 985, 1058, 1097, 1121, 1145})
 // Kernel(1) END
 // end: generate({1})
 // CodeGeneratorVisitor::generate() end
s_endpgm  // End of RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.LRR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end:
.size RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2, .LRR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end-RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
 // Resource limits
  .amdhsa_next_free_vgpr 84
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 8448 // lds bytes
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
    DataType: B8
    DataTypeA: B8
    DataTypeB: F8
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
  - .name: RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
    .symbol: RR_GEMM_TN_BF8_FP8_Half_Half_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 8448
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 58
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP8, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP8, pointerType: PointerGlobal}
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

