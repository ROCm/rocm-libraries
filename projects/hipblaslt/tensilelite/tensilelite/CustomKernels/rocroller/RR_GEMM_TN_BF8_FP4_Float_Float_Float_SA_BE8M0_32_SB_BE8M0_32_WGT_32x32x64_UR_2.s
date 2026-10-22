// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

 // Tensor.BF8.d2 0, (base=&0, sizes={&8 &16 }, strides={&24 &32 })
 // T_LOAD_TILED 1 Source 0
 // Tensor.FP4.d2 2, (base=&40, sizes={&48 &56 }, strides={&64 &72 })
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
 // Tensor_2_pointer: PointerGlobal: FP4(offset: 40, size: 8, read_write)
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
 // KernelArg{Tensor_2_pointer, PointerGlobal: FP4, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
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
.globl RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.p2align 8
.type RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2,@function
// Allocated Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1
// Allocated Preloaded argument block: 14 SGPRs (Value: Raw32): s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15
// Allocated Workgroup Index X: 1 SGPR (Value: UInt32): s16
// Allocated Workgroup Index Y: 1 SGPR (Value: UInt32): s17
// Allocated Packed Workitem Index: 1 VGPR (Value: Raw32): v0
// Allocated Workitem Index X: 1 VGPR (Value: UInt32): v1
// Allocated Workitem Index Y: 1 VGPR (Value: UInt32): v2
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2:
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
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_exec_begin:
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
 // KernelArg{Tensor_2_pointer, PointerGlobal: FP4, read_write, CommandArgument(Tensor_2_pointer)PG(c 0), o:32, s:8, preloaded}
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
 // Tensor_2_pointer: SGPR PointerGlobal: FP4 x 1: s[10:11]
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
 // (op 1) generate({761, 814, 867, 920, 978, 1002, 1026, 1050})
 // (op 1) NOP(761) BEGIN
 // (op 761) generate({})
 // (op 761) end: generate({})
 // (op 1) NOP(761) END
 // (op 1) NOP(814) BEGIN
 // (op 814) generate({})
 // (op 814) end: generate({})
 // (op 1) NOP(814) END
 // (op 1) NOP(867) BEGIN
 // (op 867) generate({})
 // (op 867) end: generate({})
 // (op 1) NOP(867) END
 // (op 1) NOP(920) BEGIN
 // (op 920) generate({})
 // (op 920) end: generate({})
 // (op 1) NOP(920) END
 // (op 1) NOP(978) BEGIN
 // (op 978) generate({})
 // (op 978) end: generate({})
 // (op 1) NOP(978) END
 // (op 1) NOP(1002) BEGIN
 // (op 1002) generate({})
 // (op 1002) end: generate({})
 // (op 1) NOP(1002) END
 // (op 1) NOP(1026) BEGIN
 // (op 1026) generate({})
 // (op 1026) end: generate({})
 // (op 1) NOP(1026) END
 // (op 1) NOP(1050) BEGIN
 // (op 1050) generate({})
 // (op 1050) end: generate({})
 // (op 1) NOP(1050) END
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(765) BEGIN
 // (op 765) Assign dim(584) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 584: v**UNALLOCATED**
 // (op 765) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag584: VGPR Value: UInt32 x 1: (unallocated)
 // (op 765) reg expression
 // (op 765) BEGIN: Flatten[1]
 // (op 765) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR LogicalShiftR(ShiftL({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32(818) BEGIN
 // (op 818) Assign dim(600) = LogicalShiftR(ShiftL({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32
 // tag 600: v**UNALLOCATED**
 // (op 818) Generate LogicalShiftR(ShiftL({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32 into DataFlowTag600: VGPR Value: UInt32 x 1: (unallocated)
 // (op 818) reg expression
 // (op 818) BEGIN: Flatten[1]
 // (op 818) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}(871) BEGIN
 // (op 871) Assign dim(616) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 616: v**UNALLOCATED**
 // (op 871) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag616: VGPR Value: UInt32 x 1: (unallocated)
 // (op 871) reg expression
 // (op 871) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}(924) BEGIN
 // (op 924) Assign dim(632) = {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 632: v**UNALLOCATED**
 // (op 924) Generate {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag632: VGPR Value: UInt32 x 1: (unallocated)
 // (op 924) reg expression
 // (op 924) BEGIN: Flatten[1]Tile
 // (op 924) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(982) BEGIN
 // (op 982) Assign dim(650) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}
 // tag 650: v**UNALLOCATED**
 // (op 982) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32} into DataFlowTag650: VGPR Value: UInt32 x 1: (unallocated)
 // (op 982) reg expression
 // (op 982) BEGIN: Flatten[1]
 // (op 982) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR LogicalShiftR(ShiftL({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32(1006) BEGIN
 // (op 1006) Assign dim(658) = LogicalShiftR(ShiftL({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32
 // tag 658: v**UNALLOCATED**
 // (op 1006) Generate LogicalShiftR(ShiftL({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32 into DataFlowTag658: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1006) reg expression
 // (op 1006) BEGIN: Flatten[1]
 // (op 1006) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}(1030) BEGIN
 // (op 1030) Assign dim(666) = {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 666: v**UNALLOCATED**
 // (op 1030) Generate {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag666: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1030) reg expression
 // (op 1030) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}(1054) BEGIN
 // (op 1054) Assign dim(674) = {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}
 // tag 674: v**UNALLOCATED**
 // (op 1054) Generate {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32} into DataFlowTag674: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1054) reg expression
 // (op 1054) BEGIN: Flatten[1]Tile
 // (op 1054) {Flatten[1]Tile: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 765): v0
v_and_b32 v0, 63, v1 // (op 765) 
 // (op 765) END: Flatten[1]
 // (op 765) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 765): v3
v_lshrrev_b32 v3, 4, v0 // (op 765) 
 // (op 765) BEGIN: Flatten[1]
 // (op 765) {Flatten[1]: BitwiseAnd(v3:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 765): v4
v_and_b32 v4, 1, v3 // (op 765) 
 // (op 765) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 765): v3

 // (op 765) BEGIN: Flatten[1]
 // (op 765) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 765): v3
v_and_b32 v3, 15, v1 // (op 765) 
 // (op 765) END: Flatten[1]
 // (op 765) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 765): v5
v_lshl_add_u32 v5, v4, 4, v3 // (op 765) 
 // (op 765) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 765): v3

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 765): v4

 // (op 765) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 765): v3
v_lshrrev_b32 v3, 6, v1 // (op 765) 
 // (op 765) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 765): v4
v_lshl_add_u32 v4, v3, 5, v5 // (op 765) 
 // (op 765) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 765): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 765): v3

 // (op 765) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 765): v3
v_lshrrev_b32 v3, 5, v0 // (op 765) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 765): v0

 // (op 765) BEGIN: TileTile
 // (op 765) {TileTile: ShiftL(v3:U32, 4:U32)U32}
 // (op 765) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 765)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 765)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 765): v0
v_lshlrev_b32 v0, 4, v3 // (op 765) 
 // (op 765) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 765): v3

 // (op 765) BEGIN: Tile
// Allocated DataFlowTag584: 1 VGPR (Value: UInt32) (op 765): v3
v_lshl_add_u32 v3, v4, 6, v0 // (op 765) 
 // (op 765) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 765): v0

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 765): v4

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(765) END
// Allocated : 1 VGPR (Value: UInt32) (op 818): v0
v_and_b32 v0, 63, v1 // (op 818) 
 // (op 818) END: Flatten[1]
 // (op 818) LogicalShiftR({Flatten[1]: v0:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 818): v4
v_lshrrev_b32 v4, 4, v0 // (op 818) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 818): v0

 // (op 818) BEGIN: Flatten[1]
 // (op 818) {Flatten[1]: BitwiseAnd(v4:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 818): v0
v_and_b32 v0, 1, v4 // (op 818) 
 // (op 818) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 818): v4

 // (op 818) BEGIN: Flatten[1]
 // (op 818) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 818): v4
v_and_b32 v4, 15, v1 // (op 818) 
 // (op 818) END: Flatten[1]
 // (op 818) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 818): v5
v_lshl_add_u32 v5, v0, 4, v4 // (op 818) 
 // (op 818) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 818): v4

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 818): v0

 // (op 818) BEGIN: TileTile
 // (op 818) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 32:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 818): v0
v_and_b32 v0, 32, v1 // (op 818) 
 // (op 818) END: TileTile
 // (op 818) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 818): v4
v_lshl_add_u32 v4, v5, 6, v0 // (op 818) 
 // (op 818) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 818): v0

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 818): v5

 // (op 818) ShiftL({Tile: v4:U32}, 2:U32)U32
 // (op 818) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 818)         value (Tile: VGPR Value: UInt32 x 1: v4) 
 // (op 818)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 818): v0
v_lshlrev_b32 v0, 2, v4 // (op 818) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 818): v4

 // (op 818) LogicalShiftR(v0:U32, 3:U32)U32
// Allocated DataFlowTag600: 1 VGPR (Value: UInt32) (op 818): v4
v_lshrrev_b32 v4, 3, v0 // (op 818) 
 // Freeing : 1 VGPR (Value: UInt32) (op 818): v0

 // (op 1) Assign VGPR LogicalShiftR(ShiftL({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32(818) END
// Allocated : 1 VGPR (Value: UInt32) (op 871): v0
v_lshrrev_b32 v0, 6, v1 // (op 871) 
 // (op 871) BEGIN: Flatten[1]
 // (op 871) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 871): v5
v_and_b32 v5, 31, v1 // (op 871) 
 // (op 871) END: Flatten[1]
 // (op 871) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 871): v6
v_lshl_add_u32 v6, v0, 5, v5 // (op 871) 
 // (op 871) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 871): v5

 // Freeing : 1 VGPR (Value: UInt32) (op 871): v0

 // (op 871) BEGIN: Flatten[1]
 // (op 871) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 871): v0
v_and_b32 v0, 63, v1 // (op 871) 
 // (op 871) END: Flatten[1]
 // (op 871) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 871): v5
v_lshrrev_b32 v5, 5, v0 // (op 871) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 871): v0

 // (op 871) BEGIN: Tile
// Allocated DataFlowTag616: 1 VGPR (Value: UInt32) (op 871): v0
v_lshl_add_u32 v0, v6, 1, v5 // (op 871) 
 // (op 871) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 871): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 871): v6

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}(871) END
// Allocated : 1 VGPR (Value: UInt32) (op 924): v5
v_and_b32 v5, 31, v1 // (op 924) 
 // (op 924) END: Flatten[1]Tile
 // (op 924) BEGIN: Flatten[1]
 // (op 924) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 924): v6
v_and_b32 v6, 63, v1 // (op 924) 
 // (op 924) END: Flatten[1]
 // (op 924) LogicalShiftR({Flatten[1]: v6:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 924): v7
v_lshrrev_b32 v7, 5, v6 // (op 924) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 924): v6

 // (op 924) BEGIN: Tile
// Allocated DataFlowTag632: 1 VGPR (Value: UInt32) (op 924): v6
v_lshl_add_u32 v6, v5, 1, v7 // (op 924) 
 // (op 924) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 924): v7

 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 924): v5

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}(924) END
// Allocated : 1 VGPR (Value: UInt32) (op 982): v5
v_and_b32 v5, 63, v1 // (op 982) 
 // (op 982) END: Flatten[1]
 // (op 982) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 982): v7
v_lshrrev_b32 v7, 4, v5 // (op 982) 
 // (op 982) BEGIN: Flatten[1]
 // (op 982) {Flatten[1]: BitwiseAnd(v7:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 982): v8
v_and_b32 v8, 1, v7 // (op 982) 
 // (op 982) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 982): v7

 // (op 982) BEGIN: Flatten[1]
 // (op 982) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 982): v7
v_and_b32 v7, 15, v1 // (op 982) 
 // (op 982) END: Flatten[1]
 // (op 982) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 982): v9
v_lshl_add_u32 v9, v8, 4, v7 // (op 982) 
 // (op 982) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 982): v7

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 982): v8

 // (op 982) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 982): v7
v_lshrrev_b32 v7, 6, v1 // (op 982) 
 // (op 982) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 982): v8
v_lshl_add_u32 v8, v7, 5, v9 // (op 982) 
 // (op 982) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 982): v9

 // Freeing : 1 VGPR (Value: UInt32) (op 982): v7

 // (op 982) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 982): v7
v_lshrrev_b32 v7, 5, v5 // (op 982) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 982): v5

 // (op 982) BEGIN: TileTile
 // (op 982) {TileTile: ShiftL(v7:U32, 4:U32)U32}
 // (op 982) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 982)         value (VGPR Value: UInt32 x 1: v7) 
 // (op 982)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 982): v5
v_lshlrev_b32 v5, 4, v7 // (op 982) 
 // (op 982) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 982): v7

 // (op 982) BEGIN: Tile
// Allocated DataFlowTag650: 1 VGPR (Value: UInt32) (op 982): v7
v_lshl_add_u32 v7, v8, 6, v5 // (op 982) 
 // (op 982) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 982): v5

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 982): v8

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(102)U32, 6:U32)U32, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 15:U32)U32})U32})U32}, 6:U32, {TileTile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(102)U32, 63:U32)U32}, 5:U32)U32, 4:U32)U32})U32}(982) END
// Allocated : 1 VGPR (Value: UInt32) (op 1006): v5
v_and_b32 v5, 63, v1 // (op 1006) 
 // (op 1006) END: Flatten[1]
 // (op 1006) LogicalShiftR({Flatten[1]: v5:U32}, 4:U32)U32
 // (op 1) Deallocate{}(1282) BEGIN
 // (op 1282) Deallocate 102
 // Deleting tag 102
 // (op 1) Deallocate{}(1282) END
// Allocated : 1 VGPR (Value: UInt32) (op 1006): v8
v_lshrrev_b32 v8, 4, v5 // (op 1006) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1006): v5

 // (op 1006) BEGIN: Flatten[1]
 // (op 1006) {Flatten[1]: BitwiseAnd(v8:U32, 1:U32)U32}
 // (op 1) Assign VGPR 64:U32(766) BEGIN
 // (op 766) Assign dim(585) = 64:U32
 // (op 766) Assign stride expression
 // (op 1) Assign VGPR 64:U32(766) END
 // (op 1) Assign VGPR 64:U32(983) BEGIN
 // (op 983) Assign dim(651) = 64:U32
 // (op 983) Assign stride expression
 // (op 1) Assign VGPR 64:U32(983) END
 // (op 1) NOP(762) BEGIN
 // (op 762) generate({})
 // (op 762) end: generate({})
 // (op 1) NOP(762) END
 // (op 1) NOP(979) BEGIN
 // (op 979) generate({})
 // (op 979) end: generate({})
 // (op 1) NOP(979) END
 // (op 1) Assign VGPR 1:U32(770) BEGIN
 // (op 770) Assign dim(587) = 1:U32
 // (op 770) Assign stride expression
 // (op 1) Assign VGPR 1:U32(770) END
 // (op 1) Assign VGPR 1:U32(987) BEGIN
 // (op 987) Assign dim(653) = 1:U32
 // (op 987) Assign stride expression
 // (op 1) Assign VGPR 1:U32(987) END
// Allocated : 1 VGPR (Value: UInt32) (op 1006): v5
v_and_b32 v5, 1, v8 // (op 1006) 
 // (op 1006) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 1006): v8

 // (op 1006) BEGIN: Flatten[1]
 // (op 1006) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1006): v8
v_and_b32 v8, 15, v1 // (op 1006) 
 // (op 1006) END: Flatten[1]
 // (op 1006) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 1006): v9
v_lshl_add_u32 v9, v5, 4, v8 // (op 1006) 
 // (op 1006) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1006): v8

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1006): v5

 // (op 1006) BEGIN: TileTile
 // (op 1006) {TileTile: BitwiseAnd({Workitem Index X: v1:U32}, 32:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1006): v5
v_and_b32 v5, 32, v1 // (op 1006) 
 // (op 1006) END: TileTile
 // (op 1006) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 1006): v8
v_lshl_add_u32 v8, v9, 6, v5 // (op 1006) 
 // (op 1006) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1006): v5

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 1006): v9

 // (op 1006) ShiftL({Tile: v8:U32}, 2:U32)U32
 // (op 1006) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 1006)         value (Tile: VGPR Value: UInt32 x 1: v8) 
 // (op 1006)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 1006): v5
v_lshlrev_b32 v5, 2, v8 // (op 1006) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1006): v8

 // (op 1006) LogicalShiftR(v5:U32, 3:U32)U32
// Allocated DataFlowTag658: 1 VGPR (Value: UInt32) (op 1006): v8
v_lshrrev_b32 v8, 3, v5 // (op 1006) 
 // Freeing : 1 VGPR (Value: UInt32) (op 1006): v5

 // (op 1) Assign VGPR LogicalShiftR(ShiftL({Tile: ShiftLAdd({TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4:U32)U32, 1:U32)U32}, 4:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 15:U32)U32})U32}, 6:U32, {TileTile: BitwiseAnd({Flatten[1]: BitwiseAnd(DataFlowTag(168)U32, 63:U32)U32}, 4294967264:U32)U32})U32}, 2:U32)U32, 3:U32)U32(1006) END
// Allocated : 1 VGPR (Value: UInt32) (op 1030): v5
v_lshrrev_b32 v5, 6, v1 // (op 1030) 
 // (op 1030) BEGIN: Flatten[1]
 // (op 1030) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 1) Deallocate{}(1285) BEGIN
 // (op 1285) Deallocate 168
 // Deleting tag 168
 // (op 1) Deallocate{}(1285) END
// Allocated : 1 VGPR (Value: UInt32) (op 1030): v9
v_and_b32 v9, 31, v1 // (op 1030) 
 // (op 1030) END: Flatten[1]
 // (op 1030) BEGIN: Tile
 // (op 1) Assign VGPR 32:U32(819) BEGIN
 // (op 819) Assign dim(601) = 32:U32
 // (op 819) Assign stride expression
 // (op 1) Assign VGPR 32:U32(819) END
 // (op 1) Assign VGPR 32:U32(1007) BEGIN
 // (op 1007) Assign dim(659) = 32:U32
 // (op 1007) Assign stride expression
 // (op 1) Assign VGPR 32:U32(1007) END
 // (op 1) NOP(815) BEGIN
 // (op 815) generate({})
 // (op 815) end: generate({})
 // (op 1) NOP(815) END
 // (op 1) NOP(1003) BEGIN
 // (op 1003) generate({})
 // (op 1003) end: generate({})
 // (op 1) NOP(1003) END
 // (op 1) Assign VGPR 0:U32(823) BEGIN
 // (op 823) Assign dim(603) = 0:U32
 // (op 823) Assign stride expression
 // (op 1) Assign VGPR 0:U32(823) END
 // (op 1) Assign VGPR 0:U32(1011) BEGIN
 // (op 1011) Assign dim(661) = 0:U32
 // (op 1011) Assign stride expression
 // (op 1) Assign VGPR 0:U32(1011) END
// Allocated : 1 VGPR (Value: UInt32) (op 1030): v10
v_lshl_add_u32 v10, v5, 5, v9 // (op 1030) 
 // (op 1030) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1030): v9

 // Freeing : 1 VGPR (Value: UInt32) (op 1030): v5

 // (op 1030) BEGIN: Flatten[1]
 // (op 1030) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1030): v5
v_and_b32 v5, 63, v1 // (op 1030) 
 // (op 1030) END: Flatten[1]
 // (op 1030) LogicalShiftR({Flatten[1]: v5:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1030): v9
v_lshrrev_b32 v9, 5, v5 // (op 1030) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1030): v5

 // (op 1030) BEGIN: Tile
// Allocated DataFlowTag666: 1 VGPR (Value: UInt32) (op 1030): v5
v_lshl_add_u32 v5, v10, 1, v9 // (op 1030) 
 // (op 1030) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 1030): v9

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 1030): v10

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(234)U32, 6:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 31:U32)U32})U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(234)U32, 63:U32)U32}, 5:U32)U32)U32}(1030) END
// Allocated : 1 VGPR (Value: UInt32) (op 1054): v9
v_and_b32 v9, 31, v1 // (op 1054) 
 // (op 1054) END: Flatten[1]Tile
 // (op 1054) BEGIN: Flatten[1]
 // (op 1054) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 1) Deallocate{}(1288) BEGIN
 // (op 1288) Deallocate 234
 // Deleting tag 234
 // (op 1) Deallocate{}(1288) END
// Allocated : 1 VGPR (Value: UInt32) (op 1054): v10
v_and_b32 v10, 63, v1 // (op 1054) 
 // (op 1054) END: Flatten[1]
 // (op 1054) LogicalShiftR({Flatten[1]: v10:U32}, 5:U32)U32
 // (op 1) Assign VGPR 2:U32(872) BEGIN
 // (op 872) Assign dim(617) = 2:U32
 // (op 872) Assign stride expression
 // (op 1) Assign VGPR 2:U32(872) END
 // (op 1) Assign VGPR 2:U32(1031) BEGIN
 // (op 1031) Assign dim(667) = 2:U32
 // (op 1031) Assign stride expression
 // (op 1) Assign VGPR 2:U32(1031) END
 // (op 1) NOP(868) BEGIN
 // (op 868) generate({})
 // (op 868) end: generate({})
 // (op 1) NOP(868) END
 // (op 1) NOP(1027) BEGIN
 // (op 1027) generate({})
 // (op 1027) end: generate({})
 // (op 1) NOP(1027) END
 // (op 1) Assign VGPR 1:U32(876) BEGIN
 // (op 876) Assign dim(619) = 1:U32
 // (op 876) Assign stride expression
 // (op 1) Assign VGPR 1:U32(876) END
 // (op 1) Assign VGPR 1:U32(1035) BEGIN
 // (op 1035) Assign dim(669) = 1:U32
 // (op 1035) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1035) END
// Allocated : 1 VGPR (Value: UInt32) (op 1054): v11
v_lshrrev_b32 v11, 5, v10 // (op 1054) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 1054): v10

 // (op 1054) BEGIN: Tile
// Allocated DataFlowTag674: 1 VGPR (Value: UInt32) (op 1054): v10
v_lshl_add_u32 v10, v9, 1, v11 // (op 1054) 
 // (op 1054) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 1054): v11

 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 1054): v9

 // (op 1) Assign VGPR {Tile: ShiftLAdd({Flatten[1]Tile: BitwiseAnd(DataFlowTag(294)U32, 31:U32)U32}, 1:U32, LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(294)U32, 63:U32)U32}, 5:U32)U32)U32}(1054) END
 // (op 1) Deallocate{}(1291) BEGIN
 // (op 1291) Deallocate 294
 // Deleting tag 294
 // (op 1) Deallocate{}(1291) END
 // (op 1) Assign VGPR 2:U32(925) BEGIN
 // (op 925) Assign dim(633) = 2:U32
 // (op 925) Assign stride expression
 // (op 1) Assign VGPR 2:U32(925) END
 // (op 1) Assign VGPR 2:U32(1055) BEGIN
 // (op 1055) Assign dim(675) = 2:U32
 // (op 1055) Assign stride expression
 // (op 1) Assign VGPR 2:U32(1055) END
 // (op 1) NOP(921) BEGIN
 // (op 921) generate({})
 // (op 921) end: generate({})
 // (op 1) NOP(921) END
 // (op 1) NOP(1051) BEGIN
 // (op 1051) generate({})
 // (op 1051) end: generate({})
 // (op 1) NOP(1051) END
 // (op 1) Assign VGPR 1:U32(929) BEGIN
 // (op 929) Assign dim(635) = 1:U32
 // (op 929) Assign stride expression
 // (op 1) Assign VGPR 1:U32(929) END
 // (op 1) Assign VGPR 1:U32(1059) BEGIN
 // (op 1059) Assign dim(677) = 1:U32
 // (op 1059) Assign stride expression
 // (op 1) Assign VGPR 1:U32(1059) END
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
 // (op 1) Deallocate{user_Float_Value_14}(1343) BEGIN
 // (op 1343) Deallocate user_Float_Value_14
 // Freeing user_Float_Value_14: 1 SGPR (Value: Raw32): s14

 // (op 1) Deallocate{user_Float_Value_14}(1343) END
 // (op 1) Deallocate{user_Float_Value_12}(1345) BEGIN
 // (op 1345) Deallocate user_Float_Value_12
 // Freeing user_Float_Value_12: 1 SGPR (Value: Raw32): s15

 // (op 1) Deallocate{user_Float_Value_12}(1345) END
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
 // (op 340) Scope(467) BEGIN
 // (op 467) Lock Scope 467
 // Lock Branch
 // (op 467) generate({462})
 // (op 467) NOP(462) BEGIN
 // (op 462) generate({})
 // (op 462) end: generate({})
 // (op 467) NOP(462) END
 // (op 467) Assign VGPR 64:U64(472) BEGIN
 // (op 472) Assign dim(498) = 64:U64
 // (op 472) Assign stride expression
 // (op 467) Assign VGPR 64:U64(472) END
 // (op 467) NOP(463) BEGIN
 // (op 463) generate({})
 // (op 463) end: generate({})
 // (op 467) NOP(463) END
 // (op 467) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, {Tile: ShiftL(LogicalShiftR(DataFlowTag(76)U32, 2:I)U32, 1:U32)U32})U32}, Convert(Tensor_0_stride_0_0:I64)U32, {TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32}(475) BEGIN
 // (op 475) Assign dim(499) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, {Tile: ShiftL(LogicalShiftR(DataFlowTag(76)U32, 2:I)U32, 1:U32)U32})U32}, Convert(Tensor_0_stride_0_0:I64)U32, {TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32}
 // tag 499: v**UNALLOCATED**
 // (op 475) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, {Tile: ShiftL(LogicalShiftR(DataFlowTag(76)U32, 2:I)U32, 1:U32)U32})U32}, Convert(Tensor_0_stride_0_0:I64)U32, {TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32} into DataFlowTag499: VGPR Value: UInt32 x 1: (unallocated)
 // (op 475) reg expression
 // (op 475) reg expression
 // (op 475) reg expression
 // (op 475) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
 // (op 475) convert
// Allocated : 1 VGPR (Value: UInt32) (op 475): v12
v_lshrrev_b32 v12, 2, v1 // (op 475) 
 // (op 475) BEGIN: Tile
 // (op 475) {Tile: ShiftL(v12:U32, 1:U32)U32}
 // (op 475) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 475)         value (VGPR Value: UInt32 x 1: v12) 
 // (op 475)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 1 VGPR (Value: UInt32) (op 475): v13
v_lshlrev_b32 v13, 1, v12 // (op 475) 
 // (op 475) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 475): v12

 // (op 475) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 475): v12
v_lshl_add_u32 v12, s16, 5, v13 // (op 475) 
 // (op 475) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 475): v13

 // (op 475) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 475): v13
v_and_b32 v13, -4, v1 // (op 475) 
 // (op 475) BEGIN: Flatten[1]
 // (op 475) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 475): v14
v_sub_u32 v14, v1, v13 // (op 475) 
 // (op 475) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 475): v13

 // (op 475) BEGIN: TileTile
 // (op 475) {TileTile: ShiftL({Flatten[1]: v14:U32}, 4:U32)U32}
 // (op 475) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 475)         value (Flatten[1]: VGPR Value: UInt32 x 1: v14) 
 // (op 475)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 475): v13
v_lshlrev_b32 v13, 4, v14 // (op 475) 
 // (op 475) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 475): v14

 // (op 475) BEGIN: Split
// Allocated DataFlowTag499: 1 VGPR (Value: UInt32) (op 475): v14
v_mul_lo_u32 v14, v12, s2 // (op 475) 
v_add_u32 v14, v14, v13 // (op 475) 
 // (op 475) END: Split
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 475): v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 475): v12

 // (op 467) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, {Tile: ShiftL(LogicalShiftR(DataFlowTag(76)U32, 2:I)U32, 1:U32)U32})U32}, Convert(Tensor_0_stride_0_0:I64)U32, {TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32}(475) END
 // (op 467) Assign VGPR Convert(Tensor_0_stride_0_0:I64)U64(476) BEGIN
 // (op 476) Assign dim(500) = Convert(Tensor_0_stride_0_0:I64)U64
 // (op 476) Assign stride expression
 // (op 467) Assign VGPR Convert(Tensor_0_stride_0_0:I64)U64(476) END
 // (op 467) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(477) BEGIN
 // (op 477) Assign dim(493) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 493: s**UNALLOCATED**
 // (op 477) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag493: SGPR Buffer: None x 1: (unallocated)
 // (op 477) reg expression
 // (op 477) reg expression
 // (op 477) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 477) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 477)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 477)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 477): s0, s1
s_mul_i32 s1, s2, s7 // (op 477) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 477): s14
s_mul_i32 s14, s3, s6 // (op 477) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 477) least significant
// Allocated : 1 SGPR (Value: Int32) (op 477): s15
s_mul_hi_u32 s15, s2, s6 // (op 477) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 477) most significant: sum
s_add_u32 s1, s1, s15 // (op 477) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 477): s15

 // Freeing : 1 SGPR (Value: Int32) (op 477): s14

 // (op 477) reg expression
 // (op 477) reg expression
// Allocated DataFlowTag493: 4 SGPRs (Buffer: None) (op 477): s40, s41, s42, s43
s_mov_b32 s40, s4 // (op 477) reg expression
s_mov_b32 s41, s5 // (op 477) reg expression
s_mov_b32 s42, s0 // (op 477) reg expression
s_mov_b32 s43, 131072 // (op 477) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 477): s0, s1

 // (op 467) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(477) END
 // (op 467) NOP(464) BEGIN
 // (op 464) generate({})
 // (op 464) end: generate({})
 // (op 467) NOP(464) END
 // (op 467) Assign VGPR 1:U64(482) BEGIN
 // (op 482) Assign dim(502) = 1:U64
 // (op 482) Assign stride expression
 // (op 467) Assign VGPR 1:U64(482) END
 // (op 467) NOP(485) BEGIN
 // (op 485) generate({})
 // (op 485) end: generate({})
 // (op 467) NOP(485) END
 // (op 467) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, {Tile: ShiftL(LogicalShiftR(DataFlowTag(76)U32, 2:I)U32, 1:U32)U32})U32}, Convert(Tensor_0_stride_0_0:I64)U32, {TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32}(494) BEGIN
 // (op 494) Assign dim(503) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, {Tile: ShiftL(LogicalShiftR(DataFlowTag(76)U32, 2:I)U32, 1:U32)U32})U32}, Convert(Tensor_0_stride_0_0:I64)U32, {TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32}
 // tag 503: v**UNALLOCATED**
 // (op 494) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, {Tile: ShiftL(LogicalShiftR(DataFlowTag(76)U32, 2:I)U32, 1:U32)U32})U32}, Convert(Tensor_0_stride_0_0:I64)U32, {TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32} into DataFlowTag503: VGPR Value: UInt32 x 1: (unallocated)
 // (op 494) reg expression
 // (op 494) reg expression
 // (op 494) reg expression
 // (op 494) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
 // (op 494) convert
// Allocated : 1 VGPR (Value: UInt32) (op 494): v12
v_lshrrev_b32 v12, 2, v1 // (op 494) 
 // (op 494) BEGIN: Tile
 // (op 494) {Tile: ShiftL(v12:U32, 1:U32)U32}
 // (op 494) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 494)         value (VGPR Value: UInt32 x 1: v12) 
 // (op 494)         shiftAmount (Literal Value: UInt32 x 0: 1)
// Allocated : 1 VGPR (Value: UInt32) (op 494): v13
v_lshlrev_b32 v13, 1, v12 // (op 494) 
 // (op 494) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 494): v12

 // (op 494) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 494): v12
v_lshl_add_u32 v12, s16, 5, v13 // (op 494) 
 // (op 494) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 494): v13

 // (op 494) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 494): v13
v_and_b32 v13, -4, v1 // (op 494) 
 // (op 494) BEGIN: Flatten[1]
 // (op 494) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 494): v15
v_sub_u32 v15, v1, v13 // (op 494) 
 // (op 494) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 494): v13

 // (op 494) BEGIN: TileTile
 // (op 494) {TileTile: ShiftL({Flatten[1]: v15:U32}, 4:U32)U32}
 // (op 494) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 494)         value (Flatten[1]: VGPR Value: UInt32 x 1: v15) 
 // (op 494)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 494): v13
v_lshlrev_b32 v13, 4, v15 // (op 494) 
 // (op 494) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 494): v15

 // (op 494) BEGIN: Split
// Allocated DataFlowTag503: 1 VGPR (Value: UInt32) (op 494): v15
v_mul_lo_u32 v15, v12, s2 // (op 494) 
v_add_u32 v15, v15, v13 // (op 494) 
 // (op 494) END: Split
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 494): v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 494): v12

 // (op 467) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, {Tile: ShiftL(LogicalShiftR(DataFlowTag(76)U32, 2:I)U32, 1:U32)U32})U32}, Convert(Tensor_0_stride_0_0:I64)U32, {TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32}(494) END
 // (op 467) Assign VGPR 64:U64(495) BEGIN
 // (op 495) Assign dim(504) = 64:U64
 // (op 495) Assign stride expression
 // (op 467) Assign VGPR 64:U64(495) END
 // (op 467) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(496) BEGIN
 // (op 496) Assign dim(493) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 496) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // (op 496) reg expression
 // (op 496) reg expression
 // (op 496) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 496) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 496)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 496)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 496): s0, s1
s_mul_i32 s1, s2, s7 // (op 496) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 496): s14
s_mul_i32 s14, s3, s6 // (op 496) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 496) least significant
// Allocated : 1 SGPR (Value: Int32) (op 496): s15
s_mul_hi_u32 s15, s2, s6 // (op 496) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 496) most significant: sum
s_add_u32 s1, s1, s15 // (op 496) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 496): s15

 // Freeing : 1 SGPR (Value: Int32) (op 496): s14

 // (op 496) reg expression
 // (op 496) reg expression
s_mov_b32 s40, s4 // (op 496) reg expression
s_mov_b32 s41, s5 // (op 496) reg expression
s_mov_b32 s42, s0 // (op 496) reg expression
s_mov_b32 s43, 131072 // (op 496) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 496): s0, s1

 // (op 467) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(496) END
 // (op 467) NOP(487) BEGIN
 // (op 487) generate({})
 // (op 487) end: generate({})
 // (op 467) NOP(487) END
 // (op 467) Assign VGPR Convert(Tensor_0_stride_0_0:I64)U64(501) BEGIN
 // (op 501) Assign dim(506) = Convert(Tensor_0_stride_0_0:I64)U64
 // (op 501) Assign stride expression
 // (op 467) Assign VGPR Convert(Tensor_0_stride_0_0:I64)U64(501) END
 // (op 467) NOP(488) BEGIN
 // (op 488) generate({})
 // (op 488) end: generate({})
 // (op 467) NOP(488) END
 // (op 467) Assign VGPR 1:U64(504) BEGIN
 // (op 504) Assign dim(508) = 1:U64
 // (op 504) Assign stride expression
 // (op 467) Assign VGPR 1:U64(504) END
 // (op 467) NOP(529) BEGIN
 // (op 529) generate({})
 // (op 529) end: generate({})
 // (op 467) NOP(529) END
 // (op 467) Assign VGPR 32:U64(536) BEGIN
 // (op 536) Assign dim(517) = 32:U64
 // (op 536) Assign stride expression
 // (op 467) Assign VGPR 32:U64(536) END
 // (op 467) NOP(530) BEGIN
 // (op 530) generate({})
 // (op 530) end: generate({})
 // (op 467) NOP(530) END
 // (op 467) Assign VGPR Convert(LogicalShiftR({Split: AddShiftL(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64, 2:U32)U64}, 3:U32)U64)U32(539) BEGIN
 // (op 539) Assign dim(518) = Convert(LogicalShiftR({Split: AddShiftL(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64, 2:U32)U64}, 3:U32)U64)U32
 // tag 518: v**UNALLOCATED**
 // (op 539) Generate Convert(LogicalShiftR({Split: AddShiftL(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64, 2:U32)U64}, 3:U32)U64)U32 into DataFlowTag518: VGPR Value: UInt32 x 1: (unallocated)
 // (op 539) reg expression
 // (op 539) reg expression
 // (op 539) reg expression
 // (op 539) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 539): v12
v_and_b32 v12, -2, v1 // (op 539) 
 // (op 539) BEGIN: Flatten[1]
 // (op 539) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v12:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 539): v13
v_sub_u32 v13, v1, v12 // (op 539) 
 // (op 539) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 539): v12

 // (op 539) BEGIN: TileTile
 // (op 539) {TileTile: ShiftL({Flatten[1]: v13:U32}, 5:U32)U32}
 // (op 539) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 539)         value (Flatten[1]: VGPR Value: UInt32 x 1: v13) 
 // (op 539)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 539): v12
v_lshlrev_b32 v12, 5, v13 // (op 539) 
 // (op 539) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 539): v13

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 539): v16, v17
v_mov_b32 v17, 0 // (op 539) convert
v_mov_b32 v16, v12 // (op 539) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 539): v12

 // (op 539) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 539): v12
v_lshrrev_b32 v12, 1, v1 // (op 539) 
 // (op 539) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 539): v13
v_lshl_add_u32 v13, s17, 5, v12 // (op 539) 
 // (op 539) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 539): v12

 // (op 539) Multiply({Tile: v13:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 539) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 539)           lhs (Tile: VGPR Value: UInt32 x 1: v13) 
 // (op 539)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 539): v18, v19
v_mul_lo_u32 v19, v13, s9 // (op 539) most significant: low of low * high
 // (op 539) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 539): v12
v_mul_hi_u32 v12, v13, s8 // (op 539) most significant: high of low * low
v_mul_lo_u32 v18, v13, s8 // (op 539) least significant: low of low * low
v_add_u32 v19, v19, v12 // (op 539) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 539): v12

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 539): v13

 // (op 539) BEGIN: Split
 // (op 539) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 539)      lhs (TileTile: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 539)      rhs (VGPR Value: Int64 x 1: v[18:19])
// Allocated : 2 VGPRs (Value: UInt64) (op 539): v12, v13
// Allocated : 2 SGPRs (Value: Bool64) (op 539): s0, s1
v_add_co_u32 v12, s[0:1], v16, v18 // (op 539) least significant half
v_addc_co_u32 v13, s[0:1], v17, v19, s[0:1] // (op 539) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 539): s0, s1

 // (op 539) ShiftL: dest (VGPR Value: UInt64 x 1: v[12:13]) = 
 // (op 539)         value (VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 539)         shiftAmount (Literal Value: UInt32 x 0: 2)
v_lshlrev_b64 v[12:13], 2, v[12:13] // (op 539) 
 // (op 539) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 539): v18, v19

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 539): v16, v17

 // (op 539) LogicalShiftR({Split: v[12:13]:U64}, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 539): v16, v17
v_lshrrev_b64 v[16:17], 3, v[12:13] // (op 539) 
 // Freeing Split: 2 VGPRs (Value: UInt64) (op 539): v12, v13

// Allocated DataFlowTag518: 1 VGPR (Value: UInt32) (op 539): v12
v_mov_b32 v12, v16 // (op 539) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 539): v16, v17

 // (op 467) Assign VGPR Convert(LogicalShiftR({Split: AddShiftL(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64, 2:U32)U64}, 3:U32)U64)U32(539) END
 // (op 467) Assign VGPR LogicalShiftR(ShiftL(Convert(Tensor_2_stride_1_3:I64)U64, 2:U32)U64, 3:U32)U64(540) BEGIN
 // (op 540) Assign dim(519) = LogicalShiftR(ShiftL(Convert(Tensor_2_stride_1_3:I64)U64, 2:U32)U64, 3:U32)U64
 // (op 540) Assign stride expression
 // (op 467) Assign VGPR LogicalShiftR(ShiftL(Convert(Tensor_2_stride_1_3:I64)U64, 2:U32)U64, 3:U32)U64(540) END
 // (op 467) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(541) BEGIN
 // (op 541) Assign dim(512) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 512: s**UNALLOCATED**
 // (op 541) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag512: SGPR Buffer: None x 1: (unallocated)
 // (op 541) reg expression
 // (op 541) reg expression
 // (op 541) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 541) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 541)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 541)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 541): s0, s1
s_mul_i32 s1, s8, s13 // (op 541) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 541): s14
s_mul_i32 s14, s9, s12 // (op 541) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 541) least significant
// Allocated : 1 SGPR (Value: Int32) (op 541): s15
s_mul_hi_u32 s15, s8, s12 // (op 541) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 541) most significant: sum
s_add_u32 s1, s1, s15 // (op 541) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 541): s15

 // Freeing : 1 SGPR (Value: Int32) (op 541): s14

 // (op 541) ShiftL(s[0:1]:I64, 2:U32)I64
 // (op 541) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 541)         value (SGPR Value: Int64 x 1: s[0:1]) 
 // (op 541)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 541): s14, s15
s_lshl_b64 s[14:15], s[0:1], 2 // (op 541) 
 // Freeing : 2 SGPRs (Value: Int64) (op 541): s0, s1

 // (op 541) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 541): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 541) 
 // Freeing : 2 SGPRs (Value: Int64) (op 541): s14, s15

 // (op 541) reg expression
 // (op 541) reg expression
// Allocated DataFlowTag512: 4 SGPRs (Buffer: None) (op 541): s44, s45, s46, s47
s_mov_b32 s44, s10 // (op 541) reg expression
s_mov_b32 s45, s11 // (op 541) reg expression
s_mov_b32 s46, s0 // (op 541) reg expression
s_mov_b32 s47, 131072 // (op 541) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 541): s0, s1

 // (op 467) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(541) END
 // (op 467) NOP(531) BEGIN
 // (op 531) generate({})
 // (op 531) end: generate({})
 // (op 467) NOP(531) END
 // (op 467) Assign VGPR 0:U64(546) BEGIN
 // (op 546) Assign dim(521) = 0:U64
 // (op 546) Assign stride expression
 // (op 467) Assign VGPR 0:U64(546) END
 // (op 467) NOP(549) BEGIN
 // (op 549) generate({})
 // (op 549) end: generate({})
 // (op 467) NOP(549) END
 // (op 467) Assign VGPR Convert(LogicalShiftR({Split: AddShiftL(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64, 2:U32)U64}, 3:U32)U64)U32(558) BEGIN
 // (op 558) Assign dim(522) = Convert(LogicalShiftR({Split: AddShiftL(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64, 2:U32)U64}, 3:U32)U64)U32
 // tag 522: v**UNALLOCATED**
 // (op 558) Generate Convert(LogicalShiftR({Split: AddShiftL(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64, 2:U32)U64}, 3:U32)U64)U32 into DataFlowTag522: VGPR Value: UInt32 x 1: (unallocated)
 // (op 558) reg expression
 // (op 558) reg expression
 // (op 558) reg expression
 // (op 558) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 558): v13
v_and_b32 v13, -2, v1 // (op 558) 
 // (op 558) BEGIN: Flatten[1]
 // (op 558) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 558): v16
v_sub_u32 v16, v1, v13 // (op 558) 
 // (op 558) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 558): v13

 // (op 558) BEGIN: TileTile
 // (op 558) {TileTile: ShiftL({Flatten[1]: v16:U32}, 5:U32)U32}
 // (op 558) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 558)         value (Flatten[1]: VGPR Value: UInt32 x 1: v16) 
 // (op 558)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 558): v13
v_lshlrev_b32 v13, 5, v16 // (op 558) 
 // (op 558) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 558): v16

// Allocated TileTile: 2 VGPRs (Value: UInt64) (op 558): v16, v17
v_mov_b32 v17, 0 // (op 558) convert
v_mov_b32 v16, v13 // (op 558) convert
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 558): v13

 // (op 558) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 558): v13
v_lshrrev_b32 v13, 1, v1 // (op 558) 
 // (op 558) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 558): v18
v_lshl_add_u32 v18, s17, 5, v13 // (op 558) 
 // (op 558) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 558): v13

 // (op 558) Multiply({Tile: v18:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 558) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 558)           lhs (Tile: VGPR Value: UInt32 x 1: v18) 
 // (op 558)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 558): v20, v21
v_mul_lo_u32 v21, v18, s9 // (op 558) most significant: low of low * high
 // (op 558) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 558): v13
v_mul_hi_u32 v13, v18, s8 // (op 558) most significant: high of low * low
v_mul_lo_u32 v20, v18, s8 // (op 558) least significant: low of low * low
v_add_u32 v21, v21, v13 // (op 558) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 558): v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 558): v18

 // (op 558) BEGIN: Split
 // (op 558) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 558)      lhs (TileTile: VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 558)      rhs (VGPR Value: Int64 x 1: v[20:21])
// Allocated : 2 VGPRs (Value: UInt64) (op 558): v18, v19
// Allocated : 2 SGPRs (Value: Bool64) (op 558): s0, s1
v_add_co_u32 v18, s[0:1], v16, v20 // (op 558) least significant half
v_addc_co_u32 v19, s[0:1], v17, v21, s[0:1] // (op 558) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 558): s0, s1

 // (op 558) ShiftL: dest (VGPR Value: UInt64 x 1: v[18:19]) = 
 // (op 558)         value (VGPR Value: UInt64 x 1: v[18:19]) 
 // (op 558)         shiftAmount (Literal Value: UInt32 x 0: 2)
v_lshlrev_b64 v[18:19], 2, v[18:19] // (op 558) 
 // (op 558) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 558): v20, v21

 // Freeing TileTile: 2 VGPRs (Value: UInt64) (op 558): v16, v17

 // (op 558) LogicalShiftR({Split: v[18:19]:U64}, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 558): v16, v17
v_lshrrev_b64 v[16:17], 3, v[18:19] // (op 558) 
 // Freeing Split: 2 VGPRs (Value: UInt64) (op 558): v18, v19

// Allocated DataFlowTag522: 1 VGPR (Value: UInt32) (op 558): v13
v_mov_b32 v13, v16 // (op 558) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 558): v16, v17

 // (op 467) Assign VGPR Convert(LogicalShiftR({Split: AddShiftL(Convert({TileTile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64, 2:U32)U64}, 3:U32)U64)U32(558) END
 // (op 467) Assign VGPR 32:U64(559) BEGIN
 // (op 559) Assign dim(523) = 32:U64
 // (op 559) Assign stride expression
 // (op 467) Assign VGPR 32:U64(559) END
 // (op 467) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(560) BEGIN
 // (op 560) Assign dim(512) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 560) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // (op 560) reg expression
 // (op 560) reg expression
 // (op 560) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 560) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 560)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 560)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 560): s0, s1
s_mul_i32 s1, s8, s13 // (op 560) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 560): s14
s_mul_i32 s14, s9, s12 // (op 560) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 560) least significant
// Allocated : 1 SGPR (Value: Int32) (op 560): s15
s_mul_hi_u32 s15, s8, s12 // (op 560) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 560) most significant: sum
s_add_u32 s1, s1, s15 // (op 560) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 560): s15

 // Freeing : 1 SGPR (Value: Int32) (op 560): s14

 // (op 560) ShiftL(s[0:1]:I64, 2:U32)I64
 // (op 560) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 560)         value (SGPR Value: Int64 x 1: s[0:1]) 
 // (op 560)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 560): s14, s15
s_lshl_b64 s[14:15], s[0:1], 2 // (op 560) 
 // Freeing : 2 SGPRs (Value: Int64) (op 560): s0, s1

 // (op 560) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 560): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 560) 
 // Freeing : 2 SGPRs (Value: Int64) (op 560): s14, s15

 // (op 560) reg expression
 // (op 560) reg expression
s_mov_b32 s44, s10 // (op 560) reg expression
s_mov_b32 s45, s11 // (op 560) reg expression
s_mov_b32 s46, s0 // (op 560) reg expression
s_mov_b32 s47, 131072 // (op 560) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 560): s0, s1

 // (op 467) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(560) END
 // (op 467) NOP(551) BEGIN
 // (op 551) generate({})
 // (op 551) end: generate({})
 // (op 467) NOP(551) END
 // (op 467) Assign VGPR LogicalShiftR(ShiftL(Convert(Tensor_2_stride_1_3:I64)U64, 2:U32)U64, 3:U32)U64(565) BEGIN
 // (op 565) Assign dim(525) = LogicalShiftR(ShiftL(Convert(Tensor_2_stride_1_3:I64)U64, 2:U32)U64, 3:U32)U64
 // (op 565) Assign stride expression
 // (op 467) Assign VGPR LogicalShiftR(ShiftL(Convert(Tensor_2_stride_1_3:I64)U64, 2:U32)U64, 3:U32)U64(565) END
 // (op 467) NOP(552) BEGIN
 // (op 552) generate({})
 // (op 552) end: generate({})
 // (op 467) NOP(552) END
 // (op 467) Assign VGPR 0:U64(568) BEGIN
 // (op 568) Assign dim(527) = 0:U64
 // (op 568) Assign stride expression
 // (op 467) Assign VGPR 0:U64(568) END
 // (op 467) NOP(593) BEGIN
 // (op 593) generate({})
 // (op 593) end: generate({})
 // (op 467) NOP(593) END
 // (op 467) Assign VGPR 2:U64(600) BEGIN
 // (op 600) Assign dim(536) = 2:U64
 // (op 600) Assign stride expression
 // (op 467) Assign VGPR 2:U64(600) END
 // (op 467) NOP(594) BEGIN
 // (op 594) generate({})
 // (op 594) end: generate({})
 // (op 467) NOP(594) END
 // (op 467) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}(603) BEGIN
 // (op 603) Assign dim(537) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}
 // tag 537: v**UNALLOCATED**
 // (op 603) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32} into DataFlowTag537: VGPR Value: UInt32 x 1: (unallocated)
 // (op 603) reg expression
 // (op 603) reg expression
 // (op 603) reg expression
 // (op 603) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 603) convert
// Allocated : 1 VGPR (Value: UInt32) (op 603): v16
v_lshrrev_b32 v16, 1, v1 // (op 603) 
 // (op 603) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 603): v17
v_lshl_add_u32 v17, s16, 5, v16 // (op 603) 
 // (op 603) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 603): v16

 // (op 603) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 603): v16
v_and_b32 v16, -2, v1 // (op 603) 
 // (op 603) BEGIN: Flatten[1]TileTile
 // (op 603) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 603): v18
v_sub_u32 v18, v1, v16 // (op 603) 
 // (op 603) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 603): v16

 // (op 603) BEGIN: Split
 // Freeing Manually loaded argument block for range 56-128: 18 SGPRs (Value: Raw32): s20, s21, s22, s23, s24, s25, s26, s27, s28, s29, s30, s31, s32, s33, s34, s35, s36, s37

// Allocated DataFlowTag537: 1 VGPR (Value: UInt32) (op 603): v16
s_waitcnt lgkmcnt(0)
v_mul_lo_u32 v16, v17, s20 // (op 603) 
v_add_u32 v16, v16, v18 // (op 603) 
 // (op 603) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 603): v18

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 603): v17

 // (op 467) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}(603) END
 // (op 467) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(604) BEGIN
 // (op 604) Assign dim(538) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 604) Assign stride expression
 // (op 467) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(604) END
 // (op 467) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(605) BEGIN
 // (op 605) Assign dim(531) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 531: s**UNALLOCATED**
 // (op 605) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag531: SGPR Buffer: None x 1: (unallocated)
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
// Allocated DataFlowTag531: 4 SGPRs (Buffer: None) (op 605): s48, s49, s50, s51
s_mov_b32 s48, s22 // (op 605) reg expression
s_mov_b32 s49, s23 // (op 605) reg expression
s_mov_b32 s50, s0 // (op 605) reg expression
s_mov_b32 s51, 131072 // (op 605) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 605): s0, s1

 // (op 467) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(605) END
 // (op 467) NOP(595) BEGIN
 // (op 595) generate({})
 // (op 595) end: generate({})
 // (op 467) NOP(595) END
 // (op 467) Assign VGPR 1:U64(610) BEGIN
 // (op 610) Assign dim(540) = 1:U64
 // (op 610) Assign stride expression
 // (op 467) Assign VGPR 1:U64(610) END
 // (op 467) NOP(613) BEGIN
 // (op 613) generate({})
 // (op 613) end: generate({})
 // (op 467) NOP(613) END
 // (op 467) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}(622) BEGIN
 // (op 622) Assign dim(541) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}
 // tag 541: v**UNALLOCATED**
 // (op 622) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32} into DataFlowTag541: VGPR Value: UInt32 x 1: (unallocated)
 // (op 622) reg expression
 // (op 622) reg expression
 // (op 622) reg expression
 // (op 622) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 622) convert
// Allocated : 1 VGPR (Value: UInt32) (op 622): v17
v_lshrrev_b32 v17, 1, v1 // (op 622) 
 // (op 622) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 622): v18
v_lshl_add_u32 v18, s16, 5, v17 // (op 622) 
 // (op 622) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 622): v17

 // (op 622) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 622): v17
v_and_b32 v17, -2, v1 // (op 622) 
 // (op 622) BEGIN: Flatten[1]TileTile
 // (op 622) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v17:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 622): v19
v_sub_u32 v19, v1, v17 // (op 622) 
 // (op 622) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 622): v17

 // (op 622) BEGIN: Split
// Allocated DataFlowTag541: 1 VGPR (Value: UInt32) (op 622): v17
v_mul_lo_u32 v17, v18, s20 // (op 622) 
v_add_u32 v17, v17, v19 // (op 622) 
 // (op 622) END: Split
 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 622): v19

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 622): v18

 // (op 467) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Flatten[1]TileTile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32}(622) END
 // (op 467) Assign VGPR 2:U64(623) BEGIN
 // (op 623) Assign dim(542) = 2:U64
 // (op 623) Assign stride expression
 // (op 467) Assign VGPR 2:U64(623) END
 // (op 467) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(624) BEGIN
 // (op 624) Assign dim(531) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 624) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // (op 624) reg expression
 // (op 624) reg expression
 // (op 624) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 624) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 624)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 624)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 624): s0, s1
s_mul_i32 s1, s20, s7 // (op 624) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 624): s14
s_mul_i32 s14, s21, s6 // (op 624) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 624) least significant
// Allocated : 1 SGPR (Value: Int32) (op 624): s15
s_mul_hi_u32 s15, s20, s6 // (op 624) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 624) most significant: sum
s_add_u32 s1, s1, s15 // (op 624) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 624): s15

 // Freeing : 1 SGPR (Value: Int32) (op 624): s14

 // (op 624) reg expression
 // (op 624) reg expression
s_mov_b32 s48, s22 // (op 624) reg expression
s_mov_b32 s49, s23 // (op 624) reg expression
s_mov_b32 s50, s0 // (op 624) reg expression
s_mov_b32 s51, 131072 // (op 624) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 624): s0, s1

 // (op 467) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(624) END
 // (op 467) NOP(615) BEGIN
 // (op 615) generate({})
 // (op 615) end: generate({})
 // (op 467) NOP(615) END
 // (op 467) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(629) BEGIN
 // (op 629) Assign dim(544) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 629) Assign stride expression
 // (op 467) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(629) END
 // (op 467) NOP(616) BEGIN
 // (op 616) generate({})
 // (op 616) end: generate({})
 // (op 467) NOP(616) END
 // (op 467) Assign VGPR 1:U64(632) BEGIN
 // (op 632) Assign dim(546) = 1:U64
 // (op 632) Assign stride expression
 // (op 467) Assign VGPR 1:U64(632) END
 // (op 467) NOP(657) BEGIN
 // (op 657) generate({})
 // (op 657) end: generate({})
 // (op 467) NOP(657) END
 // (op 467) Assign VGPR 2:U64(664) BEGIN
 // (op 664) Assign dim(555) = 2:U64
 // (op 664) Assign stride expression
 // (op 467) Assign VGPR 2:U64(664) END
 // (op 467) NOP(658) BEGIN
 // (op 658) generate({})
 // (op 658) end: generate({})
 // (op 467) NOP(658) END
 // (op 467) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(667) BEGIN
 // (op 667) Assign dim(556) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 556: v**UNALLOCATED**
 // (op 667) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag556: VGPR Value: UInt32 x 1: (unallocated)
 // (op 667) reg expression
 // (op 667) reg expression
 // (op 667) reg expression
 // (op 667) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
 // (op 667) convert
// Allocated : 1 VGPR (Value: UInt32) (op 667): v18
v_and_b32 v18, -2, v1 // (op 667) 
 // (op 667) BEGIN: Flatten[1]TileTile
 // (op 667) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v18:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 667): v19
v_sub_u32 v19, v1, v18 // (op 667) 
 // (op 667) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 667): v18

 // (op 667) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 667): v18
v_lshrrev_b32 v18, 1, v1 // (op 667) 
 // (op 667) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 667): v20
v_lshl_add_u32 v20, s17, 5, v18 // (op 667) 
 // (op 667) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 667): v18

 // (op 667) Multiply({Tile: v20:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 667): v18
v_mul_lo_u32 v18, v20, s24 // (op 667) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 667): v20

 // (op 667) BEGIN: Split
 // (op 667) {Split: Add({Flatten[1]TileTile: v19:U32}, v18:U32)U32}
// Allocated DataFlowTag556: 1 VGPR (Value: UInt32) (op 667): v20
v_add_u32 v20, v19, v18 // (op 667) 
 // (op 667) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 667): v18

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 667): v19

 // (op 467) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(667) END
 // (op 467) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(668) BEGIN
 // (op 668) Assign dim(557) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 668) Assign stride expression
 // (op 467) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(668) END
 // (op 467) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(669) BEGIN
 // (op 669) Assign dim(550) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // tag 550: s**UNALLOCATED**
 // (op 669) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag550: SGPR Buffer: None x 1: (unallocated)
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
// Allocated DataFlowTag550: 4 SGPRs (Buffer: None) (op 669): s52, s53, s54, s55
s_mov_b32 s52, s26 // (op 669) reg expression
s_mov_b32 s53, s27 // (op 669) reg expression
s_mov_b32 s54, s0 // (op 669) reg expression
s_mov_b32 s55, 131072 // (op 669) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 669): s0, s1

 // (op 467) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(669) END
 // (op 467) NOP(659) BEGIN
 // (op 659) generate({})
 // (op 659) end: generate({})
 // (op 467) NOP(659) END
 // (op 467) Assign VGPR 1:U64(674) BEGIN
 // (op 674) Assign dim(559) = 1:U64
 // (op 674) Assign stride expression
 // (op 467) Assign VGPR 1:U64(674) END
 // (op 467) NOP(677) BEGIN
 // (op 677) generate({})
 // (op 677) end: generate({})
 // (op 467) NOP(677) END
 // (op 467) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(686) BEGIN
 // (op 686) Assign dim(560) = {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 560: v**UNALLOCATED**
 // (op 686) Generate {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag560: VGPR Value: UInt32 x 1: (unallocated)
 // (op 686) reg expression
 // (op 686) reg expression
 // (op 686) reg expression
 // (op 686) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
 // (op 686) convert
// Allocated : 1 VGPR (Value: UInt32) (op 686): v18
v_and_b32 v18, -2, v1 // (op 686) 
 // (op 686) BEGIN: Flatten[1]TileTile
 // (op 686) {Flatten[1]TileTile: Subtract({Workitem Index X: v1:U32}, v18:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 686): v19
v_sub_u32 v19, v1, v18 // (op 686) 
 // (op 686) END: Flatten[1]TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 686): v18

 // (op 686) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 686): v18
v_lshrrev_b32 v18, 1, v1 // (op 686) 
 // (op 686) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 686): v21
v_lshl_add_u32 v21, s17, 5, v18 // (op 686) 
 // (op 686) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 686): v18

 // (op 686) Multiply({Tile: v21:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 686): v18
v_mul_lo_u32 v18, v21, s24 // (op 686) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 686): v21

 // (op 686) BEGIN: Split
 // (op 686) {Split: Add({Flatten[1]TileTile: v19:U32}, v18:U32)U32}
// Allocated DataFlowTag560: 1 VGPR (Value: UInt32) (op 686): v21
v_add_u32 v21, v19, v18 // (op 686) 
 // (op 686) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 686): v18

 // Freeing Flatten[1]TileTile: 1 VGPR (Value: UInt32) (op 686): v19

 // (op 467) Assign VGPR {Split: Add({Flatten[1]TileTile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(686) END
 // (op 467) Assign VGPR 2:U64(687) BEGIN
 // (op 687) Assign dim(561) = 2:U64
 // (op 687) Assign stride expression
 // (op 467) Assign VGPR 2:U64(687) END
 // (op 467) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(688) BEGIN
 // (op 688) Assign dim(550) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 688) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // (op 688) reg expression
 // (op 688) reg expression
 // (op 688) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 688) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 688)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 688)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 688): s0, s1
s_mul_i32 s1, s24, s13 // (op 688) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 688): s14
s_mul_i32 s14, s25, s12 // (op 688) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 688) least significant
// Allocated : 1 SGPR (Value: Int32) (op 688): s15
s_mul_hi_u32 s15, s24, s12 // (op 688) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 688) most significant: sum
s_add_u32 s1, s1, s15 // (op 688) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 688): s15

 // Freeing : 1 SGPR (Value: Int32) (op 688): s14

 // (op 688) reg expression
 // (op 688) reg expression
s_mov_b32 s52, s26 // (op 688) reg expression
s_mov_b32 s53, s27 // (op 688) reg expression
s_mov_b32 s54, s0 // (op 688) reg expression
s_mov_b32 s55, 131072 // (op 688) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 688): s0, s1

 // (op 467) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(688) END
 // (op 467) NOP(679) BEGIN
 // (op 679) generate({})
 // (op 679) end: generate({})
 // (op 467) NOP(679) END
 // (op 467) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(693) BEGIN
 // (op 693) Assign dim(563) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 693) Assign stride expression
 // (op 467) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(693) END
 // (op 467) NOP(680) BEGIN
 // (op 680) generate({})
 // (op 680) end: generate({})
 // (op 467) NOP(680) END
 // (op 467) Assign VGPR 1:U64(696) BEGIN
 // (op 696) Assign dim(565) = 1:U64
 // (op 696) Assign stride expression
 // (op 467) Assign VGPR 1:U64(696) END
 // (op 467) NOP(360) BEGIN
 // (op 360) generate({})
 // (op 360) end: generate({})
 // (op 467) NOP(360) END
 // (op 467) SetCoordinate(365) BEGIN
 // (op 365) generate({})
 // (op 365) end: generate({})
 // (op 365) generate({366})
 // (op 365) LoadTiled Value: FP4x8(366) BEGIN
 // (op 366) GEN: loadMacroTileVGPRCI Value: FP4x8
 // (op 366) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 366) Generate 0:U64 into nullptr
 // tag 9: v**UNALLOCATED**
 // (op 366) 	Dir = Load
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
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag518: VGPR Value: UInt32 x 1: v12
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
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 4 VGPRs (Value: FP4x8) (op 366): v22, v23, v24, v25
buffer_load_dwordx4 v[22:25], v12, s[44:47], 0 offen // (op 366) Load value
// VMEM: Expected complete at 789 (current 309)
 // (op 365) LoadTiled Value: FP4x8(366) END
 // (op 365) end: generate({366})
 // (op 467) SetCoordinate(365) END
 // (op 467) SetCoordinate(368) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag556: VGPR Value: UInt32 x 1: v20
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

// Allocated : 1 VGPR (Value: E8M0) (op 369): v18
buffer_load_ubyte v18, v20, s[52:55], 0 offen offset: 0 // (op 369) Load value
// VMEM: Expected complete at 790 (current 310)
 // (op 368) LoadTiled Value: E8M0(369) END
 // (op 368) end: generate({369})
 // (op 467) SetCoordinate(368) END
 // (op 467) SetCoordinate(371) BEGIN
 // (op 371) generate({})
 // (op 371) end: generate({})
 // (op 371) generate({372})
 // (op 371) LoadTiled Value: BF8x4(372) BEGIN
 // (op 372) GEN: loadMacroTileVGPRCI Value: BF8x4
 // (op 372) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 372) Generate Convert(Tensor_0_stride_0_0:I64)U64 into Stride500: VGPR Value: UInt64 x 1: (unallocated)
 // (op 372) reg expression
// Allocated Stride500: 2 VGPRs (Value: UInt64) (op 372): v26, v27
v_mov_b64 v[26:27], s[2:3] // (op 372) convert
 // (op 372) Generate 1:U64 into nullptr
 // tag 8: v**UNALLOCATED**
 // (op 372) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride500Tensor_0_stride_0_0: VGPR Value: UInt64 x 1: v[26:27]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(16:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(16:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 1 VGPR (Value: UInt32) (op 372): v19
v_mov_b32 v19, v14 // (op 372) 
 // (op 372)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride500Tensor_0_stride_0_0: v[26:27]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
// Allocated : 8 VGPRs (Value: BF8x4) (op 372): v28, v29, v30, v31, v32, v33, v34, v35
buffer_load_dwordx4 v[28:31], v19, s[40:43], 0 offen // (op 372) Load value
// VMEM: Expected complete at 793 (current 313)
 // (op 372) Generate Add(v19:U32, v26:R)U32 into VGPR Value: UInt32 x 1: v19
 // (op 372) reg expression
 // (op 372) reg expression
 // (op 372) Add(v19:U32, v26:R)U32
v_add_u32 v19, v19, v26 // (op 372) 
buffer_load_dwordx4 v[32:35], v19, s[40:43], 0 offen // (op 372) Load value
// VMEM: Expected stall of 475. CBNW: 0, Inc: 3
// VMEM: Expected complete at 1270 (current 790)
 // Freeing Stride500: 2 VGPRs (Value: UInt64) (op 372): v26, v27

 // Freeing : 1 VGPR (Value: UInt32) (op 372): v19

 // (op 371) LoadTiled Value: BF8x4(372) END
 // (op 371) end: generate({372})
 // (op 467) SetCoordinate(371) END
 // (op 467) SetCoordinate(374) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag537: VGPR Value: UInt32 x 1: v16
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

// Allocated : 1 VGPR (Value: E8M0) (op 375): v19
buffer_load_ubyte v19, v16, s[48:51], 0 offen offset: 0 // (op 375) Load value
// VMEM: Expected complete at 1271 (current 791)
 // (op 374) LoadTiled Value: E8M0(375) END
 // (op 374) end: generate({375})
 // (op 467) SetCoordinate(374) END
 // (op 467) SetCoordinate(377) BEGIN
 // (op 377) generate({})
 // (op 377) end: generate({})
 // (op 377) generate({378})
 // (op 377) LoadTiled Value: FP4x8(378) BEGIN
 // (op 378) GEN: loadMacroTileVGPRCI Value: FP4x8
 // (op 378) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 378) Generate 0:U64 into nullptr
 // tag 470: v**UNALLOCATED**
 // (op 378) 	Dir = Load
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
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag518: VGPR Value: UInt32 x 1: v12
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
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 4 VGPRs (Value: FP4x8) (op 378): v36, v37, v38, v39
buffer_load_dwordx4 v[36:39], v12, s[44:47], 0 offen offset:32 // (op 378) Load value
// VMEM: Expected stall of 2. CBNW: 2, Inc: 3
// VMEM: Expected complete at 1274 (current 794)
 // (op 377) LoadTiled Value: FP4x8(378) END
 // (op 377) end: generate({378})
 // (op 467) SetCoordinate(377) END
 // (op 467) Deallocate{}(1264) BEGIN
 // (op 1264) Deallocate 519
 // Deleting tag 519
 // (op 1264) Deallocate 521
 // Deleting tag 521
 // (op 1264) Deallocate 518
 // Freeing DataFlowTag518: 1 VGPR (Value: UInt32) (op 539): v12

 // Deleting tag 518
 // (op 1264) Deallocate 520
 // Deleting tag 520
 // (op 1264) Deallocate 517
 // Deleting tag 517
 // (op 467) Deallocate{}(1264) END
 // (op 467) SetCoordinate(380) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag556: VGPR Value: UInt32 x 1: v20
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

// Allocated : 1 VGPR (Value: E8M0) (op 381): v12
buffer_load_ubyte v12, v20, s[52:55], 0 offen offset:2 // (op 381) Load value
// VMEM: Expected stall of 476. CBNW: 3, Inc: 3
// VMEM: Expected complete at 1751 (current 1271)
 // (op 380) LoadTiled Value: E8M0(381) END
 // (op 380) end: generate({381})
 // (op 467) SetCoordinate(380) END
 // (op 467) Deallocate{}(1266) BEGIN
 // (op 1266) Deallocate 556
 // Freeing DataFlowTag556: 1 VGPR (Value: UInt32) (op 667): v20

 // Deleting tag 556
 // (op 1266) Deallocate 558
 // Deleting tag 558
 // (op 1266) Deallocate 555
 // Deleting tag 555
 // (op 1266) Deallocate 557
 // Deleting tag 557
 // (op 1266) Deallocate 559
 // Deleting tag 559
 // (op 467) Deallocate{}(1266) END
 // (op 467) SetCoordinate(383) BEGIN
 // (op 383) generate({})
 // (op 383) end: generate({})
 // (op 383) generate({384})
 // (op 383) LoadTiled Value: BF8x4(384) BEGIN
 // (op 384) GEN: loadMacroTileVGPRCI Value: BF8x4
 // (op 384) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 384) Generate Convert(Tensor_0_stride_0_0:I64)U64 into Stride500: VGPR Value: UInt64 x 1: (unallocated)
 // (op 384) reg expression
// Allocated Stride500: 2 VGPRs (Value: UInt64) (op 384): v26, v27
v_mov_b64 v[26:27], s[2:3] // (op 384) convert
 // (op 384) Generate 1:U64 into nullptr
 // tag 466: v**UNALLOCATED**
 // (op 384) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride500Tensor_0_stride_0_0: VGPR Value: UInt64 x 1: v[26:27]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(16:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(16:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 1 VGPR (Value: UInt32) (op 384): v20
v_mov_b32 v20, v14 // (op 384) 
 // (op 384)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride500Tensor_0_stride_0_0: v[26:27]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
// Allocated : 8 VGPRs (Value: BF8x4) (op 384): v40, v41, v42, v43, v44, v45, v46, v47
buffer_load_dwordx4 v[40:43], v20, s[40:43], 0 offen offset:64 // (op 384) Load value
// VMEM: Expected complete at 1754 (current 1274)
 // (op 384) Generate Add(v20:U32, v26:R)U32 into VGPR Value: UInt32 x 1: v20
 // (op 384) reg expression
 // (op 384) reg expression
 // (op 384) Add(v20:U32, v26:R)U32
v_add_u32 v20, v20, v26 // (op 384) 
buffer_load_dwordx4 v[44:47], v20, s[40:43], 0 offen offset:64 // (op 384) Load value
// VMEM: Expected complete at 1756 (current 1276)
 // Freeing Stride500: 2 VGPRs (Value: UInt64) (op 384): v26, v27

 // Freeing : 1 VGPR (Value: UInt32) (op 384): v20

 // (op 383) LoadTiled Value: BF8x4(384) END
 // (op 383) end: generate({384})
 // (op 467) SetCoordinate(383) END
 // (op 467) Deallocate{}(1268) BEGIN
 // (op 1268) Deallocate 502
 // Deleting tag 502
 // (op 1268) Deallocate 499
 // Freeing DataFlowTag499: 1 VGPR (Value: UInt32) (op 475): v14

 // Deleting tag 499
 // (op 1268) Deallocate 498
 // Deleting tag 498
 // (op 1268) Deallocate 500
 // Deleting tag 500
 // (op 1268) Deallocate 501
 // Deleting tag 501
 // (op 467) Deallocate{}(1268) END
 // (op 467) SetCoordinate(386) BEGIN
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
 // 	info.rowOffsetReg = DataFlowTag537: VGPR Value: UInt32 x 1: v16
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

// Allocated : 1 VGPR (Value: E8M0) (op 387): v14
buffer_load_ubyte v14, v16, s[48:51], 0 offen offset:2 // (op 387) Load value
// VMEM: Expected stall of 475. CBNW: 6, Inc: 3
// VMEM: Expected complete at 2232 (current 1752)
 // (op 386) LoadTiled Value: E8M0(387) END
 // (op 386) end: generate({387})
 // (op 467) SetCoordinate(386) END
 // (op 467) Deallocate{}(1270) BEGIN
 // (op 1270) Deallocate 537
 // Freeing DataFlowTag537: 1 VGPR (Value: UInt32) (op 603): v16

 // Deleting tag 537
 // (op 1270) Deallocate 536
 // Deleting tag 536
 // (op 1270) Deallocate 538
 // Deleting tag 538
 // (op 1270) Deallocate 540
 // Deleting tag 540
 // (op 1270) Deallocate 539
 // Deleting tag 539
 // (op 467) Deallocate{}(1270) END
 // (op 467) Scope(801) BEGIN
 // (op 801) Lock Scope 801
 // Lock Branch
 // (op 801) generate({798})
 // (op 801) NOP(798) BEGIN
 // (op 798) generate({})
 // (op 798) end: generate({})
 // (op 801) NOP(798) END
 // (op 801) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(806) BEGIN
 // (op 806) Assign dim(596) = LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32
 // tag 596: v**UNALLOCATED**
 // (op 806) Generate LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32 into DataFlowTag596: VGPR Value: UInt32 x 1: (unallocated)
 // (op 806) reg expression
 // (op 806) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 806): v16
v_and_b32 v16, -2, v1 // (op 806) 
 // (op 806) BEGIN: Tile[1]
 // (op 806) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 806): v20
v_sub_u32 v20, v1, v16 // (op 806) 
 // (op 806) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 806): v16

 // (op 806) BEGIN: Flatten
 // (op 806) {Flatten: ShiftL({Tile[1]: v20:U32}, 5:U32)U32}
 // (op 806) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 806)         value (Tile[1]: VGPR Value: UInt32 x 1: v20) 
 // (op 806)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 806): v16
v_lshlrev_b32 v16, 5, v20 // (op 806) 
 // (op 806) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 806): v20

 // (op 806) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 806): v20
v_lshrrev_b32 v20, 1, v1 // (op 806) 
 // (op 806) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 806): v26
v_lshl_add_u32 v26, v20, 6, v16 // (op 806) 
 // (op 806) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 806): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 806): v20

 // (op 806) ShiftL({Flatten: v26:U32}, 2:U32)U32
 // (op 806) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 806)         value (Flatten: VGPR Value: UInt32 x 1: v26) 
 // (op 806)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 806): v16
v_lshlrev_b32 v16, 2, v26 // (op 806) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 806): v26

 // (op 806) LogicalShiftR(v16:U32, 3:U32)U32
// Allocated DataFlowTag596: 1 VGPR (Value: UInt32) (op 806): v20
v_lshrrev_b32 v20, 3, v16 // (op 806) 
 // Freeing : 1 VGPR (Value: UInt32) (op 806): v16

 // (op 801) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(806) END
 // (op 801) Assign VGPR 32:U32(807) BEGIN
 // (op 807) Assign dim(597) = 32:U32
 // (op 807) Assign stride expression
 // (op 801) Assign VGPR 32:U32(807) END
 // (op 801) NOP(799) BEGIN
 // (op 799) generate({})
 // (op 799) end: generate({})
 // (op 801) NOP(799) END
 // (op 801) Assign VGPR 0:U32(811) BEGIN
 // (op 811) Assign dim(599) = 0:U32
 // (op 811) Assign stride expression
 // (op 801) Assign VGPR 0:U32(811) END
 // (op 801) StoreLDSTile Value: FP4x8(389) BEGIN
 // (op 389) GEN: StoreLDSTile
 // (op 389) GEN: storeMacroTileLDS OP 389 LDS 25 MacroTile 9
 // (op 389) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 389) Generate 0:U32 into nullptr
 // (op 389) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[22:25]
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

 // (op 389) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[22:25]
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
 // 	t.elementBlockStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(32:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
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
 // (op 389) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(9)
ds_write_b128 v20, v[22:25] // (op 389) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008]
// DSMEM: Expected complete at 1856 (current 1761)
// Extra dsts: LDS Value: FP4x8 x 0: LDS:(Offset: 0, Size: 1024)
 // (op 801) StoreLDSTile Value: FP4x8(389) END
 // (op 801) Deallocate{}(1272) BEGIN
 // (op 1272) Deallocate 597
 // Deleting tag 597
 // (op 1272) Deallocate 596
 // Freeing DataFlowTag596: 1 VGPR (Value: UInt32) (op 806): v20

 // Deleting tag 596
 // (op 1272) Deallocate 599
 // Deleting tag 599
 // (op 1272) Deallocate 598
 // Deleting tag 598
 // (op 801) Deallocate{}(1272) END
 // (op 801) end: generate({798})
 // (op 801) Unlock Scope 801
 // Unlock None
 // (op 467) Scope(801) END
 // (op 467) Scope(907) BEGIN
 // (op 907) Lock Scope 907
 // Lock Branch
 // (op 907) generate({904})
 // (op 907) NOP(904) BEGIN
 // (op 904) generate({})
 // (op 904) end: generate({})
 // (op 907) NOP(904) END
 // (op 907) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(912) BEGIN
 // (op 912) Assign dim(628) = {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}
 // tag 628: v**UNALLOCATED**
 // (op 912) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32} into DataFlowTag628: VGPR Value: UInt32 x 1: (unallocated)
 // (op 912) reg expression
 // (op 912) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 912): v16
v_and_b32 v16, -2, v1 // (op 912) 
 // (op 912) BEGIN: Tile[1]Flatten
 // (op 912) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 912): v20
v_sub_u32 v20, v1, v16 // (op 912) 
 // (op 912) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 912): v16

 // (op 912) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 912): v16
v_and_b32 v16, 4294967294, v1 // (op 912) 
 // (op 912) BEGIN: Flatten
 // (op 912) {Flatten: Add(v16:U32, {Tile[1]Flatten: v20:U32})U32}
// Allocated DataFlowTag628: 1 VGPR (Value: UInt32) (op 912): v26
v_add_u32 v26, v16, v20 // (op 912) 
 // (op 912) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 912): v20

 // Freeing : 1 VGPR (Value: UInt32) (op 912): v16

 // (op 907) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(912) END
 // (op 907) Assign VGPR 2:U32(913) BEGIN
 // (op 913) Assign dim(629) = 2:U32
 // (op 913) Assign stride expression
 // (op 907) Assign VGPR 2:U32(913) END
 // (op 907) NOP(905) BEGIN
 // (op 905) generate({})
 // (op 905) end: generate({})
 // (op 907) NOP(905) END
 // (op 907) Assign VGPR 1:U32(917) BEGIN
 // (op 917) Assign dim(631) = 1:U32
 // (op 917) Assign stride expression
 // (op 907) Assign VGPR 1:U32(917) END
 // (op 907) StoreLDSTile Value: E8M0(390) BEGIN
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
 // 	info.offset = Literal Value: UInt32 x 0: 1024
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
 // 	info.data = VGPR Value: E8M0 x 1: v18
 // 	info.rowOffsetReg = DataFlowTag628: VGPR Value: UInt32 x 1: v26
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
 // 	info.offset = Literal Value: UInt32 x 0: 1024
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

s_waitcnt vmcnt(8)
ds_write_b8 v26, v18 offset:1024 // (op 390) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 1862 (current 1767)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 1024, Size: 64)
 // (op 907) StoreLDSTile Value: E8M0(390) END
 // (op 907) Deallocate{}(1274) BEGIN
 // (op 1274) Deallocate 631
 // Deleting tag 631
 // (op 1274) Deallocate 630
 // Deleting tag 630
 // (op 1274) Deallocate 629
 // Deleting tag 629
 // (op 1274) Deallocate 628
 // Freeing DataFlowTag628: 1 VGPR (Value: UInt32) (op 912): v26

 // Deleting tag 628
 // (op 907) Deallocate{}(1274) END
 // (op 907) end: generate({904})
 // (op 907) Unlock Scope 907
 // Unlock None
 // (op 467) Scope(907) END
 // (op 467) Scope(748) BEGIN
 // (op 748) Lock Scope 748
 // Lock Branch
 // (op 748) generate({745})
 // (op 748) NOP(745) BEGIN
 // (op 745) generate({})
 // (op 745) end: generate({})
 // (op 748) NOP(745) END
 // (op 748) Assign VGPR {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32}(753) BEGIN
 // (op 753) Assign dim(580) = {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32}
 // tag 580: v**UNALLOCATED**
 // (op 753) Generate {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32} into DataFlowTag580: VGPR Value: UInt32 x 1: (unallocated)
 // (op 753) reg expression
 // (op 753) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 753): v16
v_and_b32 v16, -4, v1 // (op 753) 
 // (op 753) BEGIN: Tile[1]
 // (op 753) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 753): v20
v_sub_u32 v20, v1, v16 // (op 753) 
 // (op 753) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 753): v16

 // (op 753) BEGIN: Flatten
 // (op 753) {Flatten: ShiftL({Tile[1]: v20:U32}, 4:U32)U32}
 // (op 753) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 753)         value (Tile[1]: VGPR Value: UInt32 x 1: v20) 
 // (op 753)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 753): v16
v_lshlrev_b32 v16, 4, v20 // (op 753) 
 // (op 753) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 753): v20

 // (op 753) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 753): v20
v_lshrrev_b32 v20, 2, v1 // (op 753) 
 // (op 753) BEGIN: Flatten
// Allocated DataFlowTag580: 1 VGPR (Value: UInt32) (op 753): v26
v_lshl_add_u32 v26, v20, 7, v16 // (op 753) 
 // (op 753) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 753): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 753): v20

 // (op 748) Assign VGPR {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32}(753) END
 // (op 748) Assign VGPR 64:U32(754) BEGIN
 // (op 754) Assign dim(581) = 64:U32
 // (op 754) Assign stride expression
 // (op 748) Assign VGPR 64:U32(754) END
 // (op 748) NOP(746) BEGIN
 // (op 746) generate({})
 // (op 746) end: generate({})
 // (op 748) NOP(746) END
 // (op 748) Assign VGPR 1:U32(758) BEGIN
 // (op 758) Assign dim(583) = 1:U32
 // (op 758) Assign stride expression
 // (op 748) Assign VGPR 1:U32(758) END
 // (op 748) StoreLDSTile Value: BF8x4(391) BEGIN
 // (op 391) GEN: StoreLDSTile
 // (op 391) GEN: storeMacroTileLDS OP 391 LDS 17 MacroTile 8
 // (op 391) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 391) Generate 64:U32 into nullptr
 // (op 391) Generate 1:U32 into nullptr
 // (op 391) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[28:35]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 64
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1088
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

 // (op 391) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[28:35]
 // 	info.rowOffsetReg = DataFlowTag580: VGPR Value: UInt32 x 1: v26
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 64
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1088
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 391) 	info.m = 2
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 64
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(7)
ds_write_b128 v26, v[28:31] offset:1088 // (op 391) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 128, 144, 160, 176, 256, 272, 288, 304, 384, 400, 416, 432, 512, 528, 544, 560, 640, 656, 672, 688, 768, 784, 800, 816, 896, 912, 928, 944, 1024, 1040, 1056, 1072, 1152, 1168, 1184, 1200, 1280, 1296, 1312, 1328, 1408, 1424, 1440, 1456, 1536, 1552, 1568, 1584, 1664, 1680, 1696, 1712, 1792, 1808, 1824, 1840, 1920, 1936, 1952, 1968]
// DSMEM: Expected complete at 1869 (current 1774)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
s_waitcnt vmcnt(6)
ds_write_b128 v26, v[32:35] offset:1152 // (op 391) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 128, 144, 160, 176, 256, 272, 288, 304, 384, 400, 416, 432, 512, 528, 544, 560, 640, 656, 672, 688, 768, 784, 800, 816, 896, 912, 928, 944, 1024, 1040, 1056, 1072, 1152, 1168, 1184, 1200, 1280, 1296, 1312, 1328, 1408, 1424, 1440, 1456, 1536, 1552, 1568, 1584, 1664, 1680, 1696, 1712, 1792, 1808, 1824, 1840, 1920, 1936, 1952, 1968]
// DSMEM: Expected stall of 82. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 1953 (current 1858)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
 // (op 748) StoreLDSTile Value: BF8x4(391) END
 // (op 748) Deallocate{}(1276) BEGIN
 // (op 1276) Deallocate 580
 // Freeing DataFlowTag580: 1 VGPR (Value: UInt32) (op 753): v26

 // Deleting tag 580
 // (op 1276) Deallocate 581
 // Deleting tag 581
 // (op 1276) Deallocate 583
 // Deleting tag 583
 // (op 1276) Deallocate 582
 // Deleting tag 582
 // (op 748) Deallocate{}(1276) END
 // (op 748) end: generate({745})
 // (op 748) Unlock Scope 748
 // Unlock None
 // (op 467) Scope(748) END
 // (op 467) Scope(854) BEGIN
 // (op 854) Lock Scope 854
 // Lock Branch
 // (op 854) generate({851})
 // (op 854) NOP(851) BEGIN
 // (op 851) generate({})
 // (op 851) end: generate({})
 // (op 854) NOP(851) END
 // (op 854) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(859) BEGIN
 // (op 859) Assign dim(612) = {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}
 // tag 612: v**UNALLOCATED**
 // (op 859) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32} into DataFlowTag612: VGPR Value: UInt32 x 1: (unallocated)
 // (op 859) reg expression
 // (op 859) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 859): v16
v_and_b32 v16, -2, v1 // (op 859) 
 // (op 859) BEGIN: Tile[1]Flatten
 // (op 859) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 859): v20
v_sub_u32 v20, v1, v16 // (op 859) 
 // (op 859) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 859): v16

 // (op 859) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 859): v16
v_and_b32 v16, 4294967294, v1 // (op 859) 
 // (op 859) BEGIN: Flatten
 // (op 859) {Flatten: Add(v16:U32, {Tile[1]Flatten: v20:U32})U32}
// Allocated DataFlowTag612: 1 VGPR (Value: UInt32) (op 859): v26
v_add_u32 v26, v16, v20 // (op 859) 
 // (op 859) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 859): v20

 // Freeing : 1 VGPR (Value: UInt32) (op 859): v16

 // (op 854) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(859) END
 // (op 854) Assign VGPR 2:U32(860) BEGIN
 // (op 860) Assign dim(613) = 2:U32
 // (op 860) Assign stride expression
 // (op 854) Assign VGPR 2:U32(860) END
 // (op 854) NOP(852) BEGIN
 // (op 852) generate({})
 // (op 852) end: generate({})
 // (op 854) NOP(852) END
 // (op 854) Assign VGPR 1:U32(864) BEGIN
 // (op 864) Assign dim(615) = 1:U32
 // (op 864) Assign stride expression
 // (op 854) Assign VGPR 1:U32(864) END
 // (op 854) StoreLDSTile Value: E8M0(392) BEGIN
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
 // 	info.offset = Literal Value: UInt32 x 0: 3136
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
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = DataFlowTag612: VGPR Value: UInt32 x 1: v26
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
 // 	info.offset = Literal Value: UInt32 x 0: 3136
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

s_waitcnt vmcnt(5)
ds_write_b8 v26, v19 offset:3136 // (op 392) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 1959 (current 1864)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 3136, Size: 64)
 // (op 854) StoreLDSTile Value: E8M0(392) END
 // (op 854) Deallocate{}(1278) BEGIN
 // (op 1278) Deallocate 615
 // Deleting tag 615
 // (op 1278) Deallocate 612
 // Freeing DataFlowTag612: 1 VGPR (Value: UInt32) (op 859): v26

 // Deleting tag 612
 // (op 1278) Deallocate 613
 // Deleting tag 613
 // (op 1278) Deallocate 614
 // Deleting tag 614
 // (op 854) Deallocate{}(1278) END
 // (op 854) end: generate({851})
 // (op 854) Unlock Scope 854
 // Unlock None
 // (op 467) Scope(854) END
 // (op 467) NOP(363) BEGIN
 // (op 363) generate({})
 // (op 363) end: generate({})
 // (op 467) NOP(363) END
 // (op 467) NOP(720) BEGIN
 // (op 720) generate({})
 // (op 720) end: generate({})
 // (op 467) NOP(720) END
 // (op 467) NOP(717) BEGIN
 // (op 717) generate({})
 // (op 717) end: generate({})
 // (op 467) NOP(717) END
 // (op 467) Assign VGPR {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32}(724) BEGIN
 // (op 724) Assign dim(572) = {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32}
 // tag 572: v**UNALLOCATED**
 // (op 724) Generate {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32} into DataFlowTag572: VGPR Value: UInt32 x 1: (unallocated)
 // (op 724) reg expression
 // (op 724) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 724): v16
v_and_b32 v16, -4, v1 // (op 724) 
 // (op 724) BEGIN: Tile[1]
 // (op 724) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 724): v20
v_sub_u32 v20, v1, v16 // (op 724) 
 // (op 724) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 724): v16

 // (op 724) BEGIN: Flatten
 // (op 724) {Flatten: ShiftL({Tile[1]: v20:U32}, 4:U32)U32}
 // (op 724) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 724)         value (Tile[1]: VGPR Value: UInt32 x 1: v20) 
 // (op 724)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 724): v16
v_lshlrev_b32 v16, 4, v20 // (op 724) 
 // (op 724) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 724): v20

 // (op 724) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 724): v20
v_lshrrev_b32 v20, 2, v1 // (op 724) 
 // (op 724) BEGIN: Flatten
// Allocated DataFlowTag572: 1 VGPR (Value: UInt32) (op 724): v26
v_lshl_add_u32 v26, v20, 7, v16 // (op 724) 
 // (op 724) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 724): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 724): v20

 // (op 467) Assign VGPR {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32}(724) END
 // (op 467) Assign VGPR 64:U32(725) BEGIN
 // (op 725) Assign dim(573) = 64:U32
 // (op 725) Assign stride expression
 // (op 467) Assign VGPR 64:U32(725) END
 // (op 467) NOP(718) BEGIN
 // (op 718) generate({})
 // (op 718) end: generate({})
 // (op 467) NOP(718) END
 // (op 467) Assign VGPR 1:U32(729) BEGIN
 // (op 729) Assign dim(575) = 1:U32
 // (op 729) Assign stride expression
 // (op 467) Assign VGPR 1:U32(729) END
 // (op 467) NOP(772) BEGIN
 // (op 772) generate({})
 // (op 772) end: generate({})
 // (op 467) NOP(772) END
 // (op 467) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(777) BEGIN
 // (op 777) Assign dim(588) = LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32
 // tag 588: v**UNALLOCATED**
 // (op 777) Generate LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32 into DataFlowTag588: VGPR Value: UInt32 x 1: (unallocated)
 // (op 777) reg expression
 // (op 777) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 777): v16
v_and_b32 v16, -2, v1 // (op 777) 
 // (op 777) BEGIN: Tile[1]
 // (op 777) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 777): v20
v_sub_u32 v20, v1, v16 // (op 777) 
 // (op 777) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 777): v16

 // (op 777) BEGIN: Flatten
 // (op 777) {Flatten: ShiftL({Tile[1]: v20:U32}, 5:U32)U32}
 // (op 777) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 777)         value (Tile[1]: VGPR Value: UInt32 x 1: v20) 
 // (op 777)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 777): v16
v_lshlrev_b32 v16, 5, v20 // (op 777) 
 // (op 777) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 777): v20

 // (op 777) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 777): v20
v_lshrrev_b32 v20, 1, v1 // (op 777) 
 // (op 777) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 777): v27
v_lshl_add_u32 v27, v20, 6, v16 // (op 777) 
 // (op 777) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 777): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 777): v20

 // (op 777) ShiftL({Flatten: v27:U32}, 2:U32)U32
 // (op 777) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 777)         value (Flatten: VGPR Value: UInt32 x 1: v27) 
 // (op 777)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 777): v16
v_lshlrev_b32 v16, 2, v27 // (op 777) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 777): v27

 // (op 777) LogicalShiftR(v16:U32, 3:U32)U32
// Allocated DataFlowTag588: 1 VGPR (Value: UInt32) (op 777): v20
v_lshrrev_b32 v20, 3, v16 // (op 777) 
 // Freeing : 1 VGPR (Value: UInt32) (op 777): v16

 // (op 467) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(777) END
 // (op 467) Assign VGPR 32:U32(778) BEGIN
 // (op 778) Assign dim(589) = 32:U32
 // (op 778) Assign stride expression
 // (op 467) Assign VGPR 32:U32(778) END
 // (op 467) NOP(773) BEGIN
 // (op 773) generate({})
 // (op 773) end: generate({})
 // (op 467) NOP(773) END
 // (op 467) Assign VGPR 0:U32(782) BEGIN
 // (op 782) Assign dim(591) = 0:U32
 // (op 782) Assign stride expression
 // (op 467) Assign VGPR 0:U32(782) END
 // (op 467) NOP(825) BEGIN
 // (op 825) generate({})
 // (op 825) end: generate({})
 // (op 467) NOP(825) END
 // (op 467) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(830) BEGIN
 // (op 830) Assign dim(604) = {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}
 // tag 604: v**UNALLOCATED**
 // (op 830) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32} into DataFlowTag604: VGPR Value: UInt32 x 1: (unallocated)
 // (op 830) reg expression
 // (op 830) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 830): v16
v_and_b32 v16, -2, v1 // (op 830) 
 // (op 830) BEGIN: Tile[1]Flatten
 // (op 830) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 830): v27
v_sub_u32 v27, v1, v16 // (op 830) 
 // (op 830) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 830): v16

 // (op 830) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 830): v16
v_and_b32 v16, 4294967294, v1 // (op 830) 
 // (op 830) BEGIN: Flatten
 // (op 830) {Flatten: Add(v16:U32, {Tile[1]Flatten: v27:U32})U32}
// Allocated DataFlowTag604: 1 VGPR (Value: UInt32) (op 830): v48
v_add_u32 v48, v16, v27 // (op 830) 
 // (op 830) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 830): v27

 // Freeing : 1 VGPR (Value: UInt32) (op 830): v16

 // (op 467) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(830) END
 // (op 467) Assign VGPR 2:U32(831) BEGIN
 // (op 831) Assign dim(605) = 2:U32
 // (op 831) Assign stride expression
 // (op 467) Assign VGPR 2:U32(831) END
 // (op 467) NOP(826) BEGIN
 // (op 826) generate({})
 // (op 826) end: generate({})
 // (op 467) NOP(826) END
 // (op 467) Assign VGPR 1:U32(835) BEGIN
 // (op 835) Assign dim(607) = 1:U32
 // (op 835) Assign stride expression
 // (op 467) Assign VGPR 1:U32(835) END
 // (op 467) NOP(878) BEGIN
 // (op 878) generate({})
 // (op 878) end: generate({})
 // (op 467) NOP(878) END
 // (op 467) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(883) BEGIN
 // (op 883) Assign dim(620) = {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}
 // tag 620: v**UNALLOCATED**
 // (op 883) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32} into DataFlowTag620: VGPR Value: UInt32 x 1: (unallocated)
 // (op 883) reg expression
 // (op 883) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 883): v16
v_and_b32 v16, -2, v1 // (op 883) 
 // (op 883) BEGIN: Tile[1]Flatten
 // (op 883) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 883): v27
v_sub_u32 v27, v1, v16 // (op 883) 
 // (op 883) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 883): v16

 // (op 883) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 883): v16
v_and_b32 v16, 4294967294, v1 // (op 883) 
 // (op 883) BEGIN: Flatten
 // (op 883) {Flatten: Add(v16:U32, {Tile[1]Flatten: v27:U32})U32}
// Allocated DataFlowTag620: 1 VGPR (Value: UInt32) (op 883): v49
v_add_u32 v49, v16, v27 // (op 883) 
 // (op 883) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 883): v27

 // Freeing : 1 VGPR (Value: UInt32) (op 883): v16

 // (op 467) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(883) END
 // (op 467) Assign VGPR 2:U32(884) BEGIN
 // (op 884) Assign dim(621) = 2:U32
 // (op 884) Assign stride expression
 // (op 467) Assign VGPR 2:U32(884) END
 // (op 467) NOP(879) BEGIN
 // (op 879) generate({})
 // (op 879) end: generate({})
 // (op 467) NOP(879) END
 // (op 467) Assign VGPR 1:U32(888) BEGIN
 // (op 888) Assign dim(623) = 1:U32
 // (op 888) Assign stride expression
 // (op 467) Assign VGPR 1:U32(888) END
 // (op 467) NOP(965) BEGIN
 // (op 965) generate({})
 // (op 965) end: generate({})
 // (op 467) NOP(965) END
 // (op 467) Assign VGPR {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32}(970) BEGIN
 // (op 970) Assign dim(646) = {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32}
 // tag 646: v**UNALLOCATED**
 // (op 970) Generate {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32} into DataFlowTag646: VGPR Value: UInt32 x 1: (unallocated)
 // (op 970) reg expression
 // (op 970) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 970): v16
v_and_b32 v16, -4, v1 // (op 970) 
 // (op 970) BEGIN: Tile[1]
 // (op 970) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 970): v27
v_sub_u32 v27, v1, v16 // (op 970) 
 // (op 970) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 970): v16

 // (op 970) BEGIN: Flatten
 // (op 970) {Flatten: ShiftL({Tile[1]: v27:U32}, 4:U32)U32}
 // (op 970) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 970)         value (Tile[1]: VGPR Value: UInt32 x 1: v27) 
 // (op 970)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 970): v16
v_lshlrev_b32 v16, 4, v27 // (op 970) 
 // (op 970) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 970): v27

 // (op 970) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 970): v27
v_lshrrev_b32 v27, 2, v1 // (op 970) 
 // (op 970) BEGIN: Flatten
// Allocated DataFlowTag646: 1 VGPR (Value: UInt32) (op 970): v50
v_lshl_add_u32 v50, v27, 7, v16 // (op 970) 
 // (op 970) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 970): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 970): v27

 // (op 467) Assign VGPR {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32}(970) END
 // (op 467) Assign VGPR 64:U32(971) BEGIN
 // (op 971) Assign dim(647) = 64:U32
 // (op 971) Assign stride expression
 // (op 467) Assign VGPR 64:U32(971) END
 // (op 467) NOP(966) BEGIN
 // (op 966) generate({})
 // (op 966) end: generate({})
 // (op 467) NOP(966) END
 // (op 467) Assign VGPR 1:U32(975) BEGIN
 // (op 975) Assign dim(649) = 1:U32
 // (op 975) Assign stride expression
 // (op 467) Assign VGPR 1:U32(975) END
 // (op 467) NOP(989) BEGIN
 // (op 989) generate({})
 // (op 989) end: generate({})
 // (op 467) NOP(989) END
 // (op 467) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(994) BEGIN
 // (op 994) Assign dim(654) = LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32
 // tag 654: v**UNALLOCATED**
 // (op 994) Generate LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32 into DataFlowTag654: VGPR Value: UInt32 x 1: (unallocated)
 // (op 994) reg expression
 // (op 994) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 994): v16
v_and_b32 v16, -2, v1 // (op 994) 
 // (op 994) BEGIN: Tile[1]
 // (op 994) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 994): v27
v_sub_u32 v27, v1, v16 // (op 994) 
 // (op 994) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 994): v16

 // (op 994) BEGIN: Flatten
 // (op 994) {Flatten: ShiftL({Tile[1]: v27:U32}, 5:U32)U32}
 // (op 994) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 994)         value (Tile[1]: VGPR Value: UInt32 x 1: v27) 
 // (op 994)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 994): v16
v_lshlrev_b32 v16, 5, v27 // (op 994) 
 // (op 994) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 994): v27

 // (op 994) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 994): v27
v_lshrrev_b32 v27, 1, v1 // (op 994) 
 // (op 994) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 994): v51
v_lshl_add_u32 v51, v27, 6, v16 // (op 994) 
 // (op 994) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 994): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 994): v27

 // (op 994) ShiftL({Flatten: v51:U32}, 2:U32)U32
 // (op 994) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 994)         value (Flatten: VGPR Value: UInt32 x 1: v51) 
 // (op 994)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 994): v16
v_lshlrev_b32 v16, 2, v51 // (op 994) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 994): v51

 // (op 994) LogicalShiftR(v16:U32, 3:U32)U32
// Allocated DataFlowTag654: 1 VGPR (Value: UInt32) (op 994): v27
v_lshrrev_b32 v27, 3, v16 // (op 994) 
 // Freeing : 1 VGPR (Value: UInt32) (op 994): v16

 // (op 467) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(994) END
 // (op 467) Assign VGPR 32:U32(995) BEGIN
 // (op 995) Assign dim(655) = 32:U32
 // (op 995) Assign stride expression
 // (op 467) Assign VGPR 32:U32(995) END
 // (op 467) NOP(990) BEGIN
 // (op 990) generate({})
 // (op 990) end: generate({})
 // (op 467) NOP(990) END
 // (op 467) Assign VGPR 0:U32(999) BEGIN
 // (op 999) Assign dim(657) = 0:U32
 // (op 999) Assign stride expression
 // (op 467) Assign VGPR 0:U32(999) END
 // (op 467) NOP(1013) BEGIN
 // (op 1013) generate({})
 // (op 1013) end: generate({})
 // (op 467) NOP(1013) END
 // (op 467) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(1018) BEGIN
 // (op 1018) Assign dim(662) = {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}
 // tag 662: v**UNALLOCATED**
 // (op 1018) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32} into DataFlowTag662: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1018) reg expression
 // (op 1018) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1018): v16
v_and_b32 v16, -2, v1 // (op 1018) 
 // (op 1018) BEGIN: Tile[1]Flatten
 // (op 1018) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1018): v51
v_sub_u32 v51, v1, v16 // (op 1018) 
 // (op 1018) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1018): v16

 // (op 1018) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1018): v16
v_and_b32 v16, 4294967294, v1 // (op 1018) 
 // (op 1018) BEGIN: Flatten
 // (op 1018) {Flatten: Add(v16:U32, {Tile[1]Flatten: v51:U32})U32}
// Allocated DataFlowTag662: 1 VGPR (Value: UInt32) (op 1018): v52
v_add_u32 v52, v16, v51 // (op 1018) 
 // (op 1018) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 1018): v51

 // Freeing : 1 VGPR (Value: UInt32) (op 1018): v16

 // (op 467) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(1018) END
 // (op 467) Assign VGPR 2:U32(1019) BEGIN
 // (op 1019) Assign dim(663) = 2:U32
 // (op 1019) Assign stride expression
 // (op 467) Assign VGPR 2:U32(1019) END
 // (op 467) NOP(1014) BEGIN
 // (op 1014) generate({})
 // (op 1014) end: generate({})
 // (op 467) NOP(1014) END
 // (op 467) Assign VGPR 1:U32(1023) BEGIN
 // (op 1023) Assign dim(665) = 1:U32
 // (op 1023) Assign stride expression
 // (op 467) Assign VGPR 1:U32(1023) END
 // (op 467) NOP(1037) BEGIN
 // (op 1037) generate({})
 // (op 1037) end: generate({})
 // (op 467) NOP(1037) END
 // (op 467) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(1042) BEGIN
 // (op 1042) Assign dim(670) = {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}
 // tag 670: v**UNALLOCATED**
 // (op 1042) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32} into DataFlowTag670: VGPR Value: UInt32 x 1: (unallocated)
 // (op 1042) reg expression
 // (op 1042) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1042): v16
v_and_b32 v16, -2, v1 // (op 1042) 
 // (op 1042) BEGIN: Tile[1]Flatten
 // (op 1042) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 1042): v51
v_sub_u32 v51, v1, v16 // (op 1042) 
 // (op 1042) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 1042): v16

 // (op 1042) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 1042): v16
v_and_b32 v16, 4294967294, v1 // (op 1042) 
 // (op 1042) BEGIN: Flatten
 // (op 1042) {Flatten: Add(v16:U32, {Tile[1]Flatten: v51:U32})U32}
// Allocated DataFlowTag670: 1 VGPR (Value: UInt32) (op 1042): v53
v_add_u32 v53, v16, v51 // (op 1042) 
 // (op 1042) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 1042): v51

 // Freeing : 1 VGPR (Value: UInt32) (op 1042): v16

 // (op 467) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(1042) END
 // (op 467) Assign VGPR 2:U32(1043) BEGIN
 // (op 1043) Assign dim(671) = 2:U32
 // (op 1043) Assign stride expression
 // (op 467) Assign VGPR 2:U32(1043) END
 // (op 467) NOP(1038) BEGIN
 // (op 1038) generate({})
 // (op 1038) end: generate({})
 // (op 467) NOP(1038) END
 // (op 467) Assign VGPR 1:U32(1047) BEGIN
 // (op 1047) Assign dim(673) = 1:U32
 // (op 1047) Assign stride expression
 // (op 467) Assign VGPR 1:U32(1047) END
 // (op 467) ForLoopOp KLoop: LessThan(DataFlowTag(395)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL(58) BEGIN
 // (op 58) Initialize For Loop
 // (op 58) generate({59})
 // (op 58) Assign SGPR 0:U32(59) BEGIN
 // (op 59) Assign dim(395) = 0:U32
 // tag 395: s**UNALLOCATED**
 // (op 59) Generate 0:U32 into DataFlowTag395: SGPR Value: UInt32 x 1: (unallocated)
// Allocated DataFlowTag395: 1 SGPR (Value: UInt32) (op 59): s0
s_mov_b32 s0, 0 // (op 59) call()
 // (op 58) Assign SGPR 0:U32(59) END
 // (op 58) end: generate({59})
 // (op 58) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 58) 
 // (op 58) Generate LessThan(DataFlowTag(395)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 58) LessThan({DataFlowTag395: s0:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 58) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32
 // (op 58) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 58): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 58) 
// Allocated : 1 SGPR (Value: UInt32) (op 58): s1
s_mov_b32 s1, s14 // (op 58) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 58): s14, s15

// Allocated : 1 SGPR (Value: UInt32) (op 58): s14
s_and_b32 s14, s1, 4294967294 // (op 58) 
 // Freeing : 1 SGPR (Value: UInt32) (op 58): s1

s_cmp_lt_u32 s0, s14 // (op 58) 
 // Freeing : 1 SGPR (Value: UInt32) (op 58): s14

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 58) 
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 // (op 58) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58 if false)
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58:
 // (op 58) 
 // (op 58) generate({60, 288, 1091})
 // (op 58) Assign SGPR Add(DataFlowTag(395)U32, 2:U32)U32(60) BEGIN
 // (op 60) Assign dim(395) = Add(DataFlowTag(395)U32, 2:U32)U32
 // (op 60) Generate Add(DataFlowTag(395)U32, 2:U32)U32 into DataFlowTag395: SGPR Value: UInt32 x 1: s0
 // (op 60) reg expression
 // (op 60) Add({DataFlowTag395: s0:U32}, 2:U32)U32
 // (op 58) SetCoordinate(288) BEGIN
 // (op 288) generate({})
 // (op 288) end: generate({})
 // (op 288) generate({10})
 // (op 288) LoadTiled Value: FP4x8(10) BEGIN
 // (op 10) GEN: loadMacroTileVGPRCI Value: FP4x8
 // (op 10) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 10) Generate 0:U64 into nullptr
 // (op 10) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[22:25]
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
 // (op 58) Barrier(1091) BEGIN
s_add_u32 s0, s0, 2 // (op 60) 
 // (op 58) Assign SGPR Add(DataFlowTag(395)U32, 2:U32)U32(60) END
// Allocated Offset524: 1 VGPR (Value: UInt32) (op 10): v16
v_mov_b32 v16, v13 // (op 10) 
 // (op 10) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[22:25]
 // 	info.rowOffsetReg = Offset524: VGPR Value: UInt32 x 1: v16
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
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

buffer_load_dwordx4 v[22:25], v16, s[44:47], 0 offen offset:64 // (op 10) Load value
// VMEM: Expected complete at 2394 (current 1914)
 // (op 288) LoadTiled Value: FP4x8(10) END
 // (op 288) end: generate({10})
 // (op 58) SetCoordinate(288) END
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1091) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
 // (op 58) Barrier(1091) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 1088
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
 // 	info.offset = Literal Value: UInt32 x 0: 1088
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

// Allocated Offset562: 1 VGPR (Value: UInt32) (op 25): v51
v_mov_b32 v51, v21 // (op 25) 
 // (op 25) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v18
 // 	info.rowOffsetReg = Offset562: VGPR Value: UInt32 x 1: v51
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

buffer_load_ubyte v18, v51, s[52:55], 0 offen offset:4 // (op 25) Load value
// VMEM: Expected complete at 2398 (current 1918)
 // (op 298) LoadTiled Value: E8M0(25) END
 // (op 298) end: generate({25})
 // (op 58) SetCoordinate(298) END
// Allocated : 8 VGPRs (Value: BF8x4) (op 70): v54, v55, v56, v57, v58, v59, v60, v61
ds_read_b128 v[54:57], v3 offset:1088 // (op 70) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 2053 (current 1958)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
 // (op 58) SetCoordinate(283) BEGIN
 // (op 283) generate({})
 // (op 283) end: generate({})
 // (op 283) generate({4})
 // (op 283) LoadTiled Value: BF8x4(4) BEGIN
 // (op 4) GEN: loadMacroTileVGPRCI Value: BF8x4
 // (op 4) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 4) Generate Convert(Tensor_0_stride_0_0:I64)U64 into Stride506: VGPR Value: UInt64 x 1: (unallocated)
 // (op 4) reg expression
ds_read_b128 v[58:61], v3 offset:1120 // (op 70) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 2054 (current 1959)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
 // (op 68) LoadLDSTile{Value: BF8x4}(70) END
 // (op 68) end: generate({70})
 // (op 58) SetCoordinate(68) END
// Allocated Stride506: 2 VGPRs (Value: UInt64) (op 4): v62, v63
v_mov_b64 v[62:63], s[2:3] // (op 4) convert
 // (op 4) Generate 1:U64 into nullptr
 // (op 4) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[28:35]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride506Tensor_0_stride_0_0: VGPR Value: UInt64 x 1: v[62:63]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(16:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(16:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 505: v**UNALLOCATED**
 // (op 58) Barrier(1104) BEGIN
s_barrier  // (op 1104) 
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 0, Size: 1024)
 // (op 58) Barrier(1104) END
// Allocated Offset505: 1 VGPR (Value: UInt32) (op 4): v64
v_mov_b32 v64, v15 // (op 4) 
 // (op 58) SetCoordinate(73) BEGIN
 // (op 73) generate({})
 // (op 73) end: generate({})
 // (op 73) generate({75})
 // (op 73) LoadLDSTile{Value: FP4x8}(75) BEGIN
 // (op 75) GEN: loadMacroTileWAVELDS OP 75 LDS 25 WaveTile 169
 // (op 75) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 75) Generate 0:U32 into nullptr
 // tag 23: v**UNALLOCATED**
 // (op 75) 	Dir = Load
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

 // (op 75) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
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
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
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
 // (op 75) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 1 VGPR (Value: UInt32) (op 4): v65
v_mov_b32 v65, v64 // (op 4) 
 // (op 4)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride506Tensor_0_stride_0_0: v[62:63]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
// Allocated : 4 VGPRs (Value: FP4x8) (op 75): v66, v67, v68, v69
ds_read_b128 v[66:69], v4 // (op 75) Load local data 
// Modelled addresses (normalized): [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 16, 48, 80, 112, 144, 176, 208, 240, 272, 304, 336, 368, 400, 432, 464, 496, 528, 560, 592, 624, 656, 688, 720, 752, 784, 816, 848, 880, 912, 944, 976, 1008]
// DSMEM: Expected complete at 2059 (current 1964)
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 0, Size: 1024)
 // (op 73) LoadLDSTile{Value: FP4x8}(75) END
 // (op 73) end: generate({75})
 // (op 58) SetCoordinate(73) END
buffer_load_dwordx4 v[28:31], v65, s[40:43], 0 offen offset:128 // (op 4) Load value
// VMEM: Expected stall of 307. CBNW: 4, Inc: 3
// VMEM: Expected complete at 2713 (current 2233)
 // (op 4) Generate Add(v65:U32, v62:R)U32 into VGPR Value: UInt32 x 1: v65
 // (op 4) reg expression
 // (op 4) reg expression
 // (op 4) Add(v65:U32, v62:R)U32
 // (op 58) Barrier(1118) BEGIN
v_add_u32 v65, v65, v62 // (op 4) 
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1118) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 3136, Size: 64)
 // (op 58) Barrier(1118) END
buffer_load_dwordx4 v[32:35], v65, s[40:43], 0 offen offset:128 // (op 4) Load value
// VMEM: Expected stall of 158. CBNW: 5, Inc: 3
// VMEM: Expected complete at 2875 (current 2395)
 // Freeing Stride506: 2 VGPRs (Value: UInt64) (op 4): v62, v63

 // Freeing : 1 VGPR (Value: UInt32) (op 4): v65

 // (op 283) LoadTiled Value: BF8x4(4) END
 // (op 283) end: generate({4})
 // (op 58) SetCoordinate(283) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 3136
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
 // 	info.offset = Literal Value: UInt32 x 0: 3136
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
// Allocated : 1 VGPR (Value: E8M0) (op 80): v65
ds_read_u8 v65, v0 offset:3136 // (op 80) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 2530 (current 2435)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 3136, Size: 64)
 // (op 78) LoadLDSTile{Value: E8M0}(80) END
 // (op 78) end: generate({80})
 // (op 58) SetCoordinate(78) END
// Allocated Offset543: 1 VGPR (Value: UInt32) (op 17): v62
v_mov_b32 v62, v17 // (op 17) 
 // (op 17) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: v19
 // 	info.rowOffsetReg = Offset543: VGPR Value: UInt32 x 1: v62
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

 // (op 58) Barrier(1132) BEGIN
s_barrier  // (op 1132) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 1024, Size: 64)
 // (op 58) Barrier(1132) END
buffer_load_ubyte v19, v62, s[48:51], 0 offen offset:4 // (op 17) Load value
// VMEM: Expected complete at 2879 (current 2399)
 // (op 293) LoadTiled Value: E8M0(17) END
 // (op 293) end: generate({17})
 // (op 58) SetCoordinate(293) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 1024
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
 // 	info.offset = Literal Value: UInt32 x 0: 1024
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

 // (op 58) Barrier(1155) BEGIN
s_barrier  // (op 1155) 
 // (op 58) Barrier(1155) END
// Allocated : 1 VGPR (Value: E8M0) (op 85): v63
ds_read_u8 v63, v6 offset:1024 // (op 85) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 2535 (current 2440)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 1024, Size: 64)
 // (op 83) LoadLDSTile{Value: E8M0}(85) END
 // (op 83) end: generate({85})
 // (op 58) SetCoordinate(83) END
 // (op 58) StoreLDSTile Value: FP4x8(253) BEGIN
 // (op 253) GEN: StoreLDSTile
 // (op 253) GEN: storeMacroTileLDS OP 253 LDS 472 MacroTile 470
 // (op 253) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 253) Generate 0:U32 into nullptr
 // (op 253) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[36:39]
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[36:39]
 // 	info.rowOffsetReg = DataFlowTag654: VGPR Value: UInt32 x 1: v27
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 58) Multiply(88) BEGIN
 // (op 88) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile8xFP4, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)S into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[54:61], v[66:69], a[0:15], v65, v63 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 88) 
 // (op 58) Multiply(88) END
s_waitcnt vmcnt(9)
ds_write_b128 v27, v[36:39] offset:3200 // (op 253) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008]
// DSMEM: Expected complete at 2634 (current 2539)
// Extra dsts: LDS Value: FP4x8 x 0: LDS:(Offset: 3200, Size: 1024)
 // (op 58) StoreLDSTile Value: FP4x8(253) END
 // (op 58) Deallocate{}(1182) BEGIN
 // (op 1182) Deallocate 23
 // Freeing : 4 VGPRs (Value: FP4x8) (op 75): v66, v67, v68, v69

 // Deleting tag 23
 // (op 1182) Deallocate 15
 // Freeing : 8 VGPRs (Value: BF8x4) (op 70): v54, v55, v56, v57, v58, v59, v60, v61

 // Deleting tag 15
 // (op 1182) Deallocate 31
 // Freeing : 1 VGPR (Value: E8M0) (op 80): v65

 // Deleting tag 31
 // (op 1182) Deallocate 39
 // Freeing : 1 VGPR (Value: E8M0) (op 85): v63

 // Deleting tag 39
 // (op 58) Deallocate{}(1182) END
 // (op 58) Deallocate{}(1246) BEGIN
 // (op 1246) Deallocate 656
 // Deleting tag 656
 // (op 58) Deallocate{}(1246) END
 // (op 58) Barrier(1167) BEGIN
s_barrier  // (op 1167) 
 // (op 58) Barrier(1167) END
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
 // 	info.data = VGPR Value: E8M0 x 1: v12
 // 	info.rowOffsetReg = DataFlowTag670: VGPR Value: UInt32 x 1: v53
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

s_waitcnt vmcnt(8)
ds_write_b8 v53, v12 offset:4224 // (op 255) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 2637 (current 2542)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 4224, Size: 64)
 // (op 58) StoreLDSTile Value: E8M0(255) END
 // (op 58) Deallocate{}(1250) BEGIN
 // (op 1250) Deallocate 672
 // Deleting tag 672
 // (op 58) Deallocate{}(1250) END
 // (op 58) Barrier(1149) BEGIN
s_barrier  // (op 1149) 
 // (op 58) Barrier(1149) END
 // (op 58) StoreLDSTile Value: BF8x4(243) BEGIN
 // (op 243) GEN: StoreLDSTile
 // (op 243) GEN: storeMacroTileLDS OP 243 LDS 468 MacroTile 466
 // (op 243) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 243) Generate 64:U32 into nullptr
 // (op 243) Generate 1:U32 into nullptr
 // (op 243) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[40:47]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 64
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
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

 // (op 243) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[40:47]
 // 	info.rowOffsetReg = DataFlowTag646: VGPR Value: UInt32 x 1: v50
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 64
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
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
 // (op 243) 	info.m = 2
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 64
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(7)
ds_write_b128 v50, v[40:43] offset:4288 // (op 243) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 128, 144, 160, 176, 256, 272, 288, 304, 384, 400, 416, 432, 512, 528, 544, 560, 640, 656, 672, 688, 768, 784, 800, 816, 896, 912, 928, 944, 1024, 1040, 1056, 1072, 1152, 1168, 1184, 1200, 1280, 1296, 1312, 1328, 1408, 1424, 1440, 1456, 1536, 1552, 1568, 1584, 1664, 1680, 1696, 1712, 1792, 1808, 1824, 1840, 1920, 1936, 1952, 1968]
// DSMEM: Expected complete at 2640 (current 2545)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 4288, Size: 2048)
s_waitcnt vmcnt(6)
ds_write_b128 v50, v[44:47] offset:4352 // (op 243) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 128, 144, 160, 176, 256, 272, 288, 304, 384, 400, 416, 432, 512, 528, 544, 560, 640, 656, 672, 688, 768, 784, 800, 816, 896, 912, 928, 944, 1024, 1040, 1056, 1072, 1152, 1168, 1184, 1200, 1280, 1296, 1312, 1328, 1408, 1424, 1440, 1456, 1536, 1552, 1568, 1584, 1664, 1680, 1696, 1712, 1792, 1808, 1824, 1840, 1920, 1936, 1952, 1968]
// DSMEM: Expected stall of 89. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 2731 (current 2636)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 4288, Size: 2048)
 // (op 58) StoreLDSTile Value: BF8x4(243) END
 // (op 58) Deallocate{}(1218) BEGIN
 // (op 1218) Deallocate 648
 // Deleting tag 648
 // (op 58) Deallocate{}(1218) END
 // (op 58) Barrier(1161) BEGIN
s_barrier  // (op 1161) 
 // (op 58) Barrier(1161) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 6336
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
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = DataFlowTag662: VGPR Value: UInt32 x 1: v52
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
 // 	info.offset = Literal Value: UInt32 x 0: 6336
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

s_waitcnt vmcnt(5)
ds_write_b8 v52, v14 offset:6336 // (op 254) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 2734 (current 2639)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 6336, Size: 64)
 // (op 58) StoreLDSTile Value: E8M0(254) END
 // (op 58) Deallocate{}(1248) BEGIN
 // (op 1248) Deallocate 664
 // Deleting tag 664
 // (op 58) Deallocate{}(1248) END
 // (op 58) NOP(406) BEGIN
 // (op 406) generate({})
 // (op 406) end: generate({})
 // (op 58) NOP(406) END
 // (op 58) SetCoordinate(309) BEGIN
 // (op 309) generate({})
 // (op 309) end: generate({})
 // (op 309) generate({256})
 // (op 309) LoadTiled Value: FP4x8(256) BEGIN
 // (op 256) GEN: loadMacroTileVGPRCI Value: FP4x8
 // (op 256) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 256) Generate 0:U64 into nullptr
 // (op 256) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[36:39]
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
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[36:39]
 // 	info.rowOffsetReg = Offset524: VGPR Value: UInt32 x 1: v16
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
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 58) Barrier(1146) BEGIN
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1146) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 4288, Size: 2048)
 // (op 58) Barrier(1146) END
buffer_load_dwordx4 v[36:39], v16, s[44:47], 0 offen offset:96 // (op 256) Load value
// VMEM: Expected stall of 206. CBNW: 2, Inc: 3
// VMEM: Expected complete at 3194 (current 2714)
 // (op 309) LoadTiled Value: FP4x8(256) END
 // (op 309) end: generate({256})
 // (op 58) SetCoordinate(309) END
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

 // (op 245) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 8
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag650: VGPR Value: UInt32 x 1: v7
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
 // (op 245) 	info.m = 1
 // 	info.n = 8
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 2
 // 	elementBlockStride = 32

 // (op 58) Deallocate{}(1254) BEGIN
 // (op 1254) Deallocate 524
 // Freeing Offset524: 1 VGPR (Value: UInt32) (op 10): v16

 // Deleting tag 524
 // (op 1254) Deallocate 526
 // Deleting tag 526
 // (op 58) Deallocate{}(1254) END
// Allocated : 8 VGPRs (Value: BF8x4) (op 245): v54, v55, v56, v57, v58, v59, v60, v61
ds_read_b128 v[54:57], v7 offset:4288 // (op 245) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 3036 (current 2941)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 4288, Size: 2048)
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
 // 	info.data = VGPR Value: E8M0 x 1: v12
 // 	info.rowOffsetReg = Offset562: VGPR Value: UInt32 x 1: v51
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

 // (op 58) Assign VGPR Add(DataFlowTag(522)U32, ShiftL(Convert(DataFlowTag(523)U64)U32, 1:U32)U32)U32(550) BEGIN
 // (op 550) Assign dim(522) = Add(DataFlowTag(522)U32, ShiftL(Convert(DataFlowTag(523)U64)U32, 1:U32)U32)U32
 // (op 550) Generate Add(DataFlowTag(522)U32, ShiftL(Convert(DataFlowTag(523)U64)U32, 1:U32)U32)U32 into DataFlowTag522: VGPR Value: UInt32 x 1: v13
 // (op 550) reg expression
 // (op 550) Add({DataFlowTag522: v13:U32}, 64:U32)U32
ds_read_b128 v[58:61], v7 offset:4320 // (op 245) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 3037 (current 2942)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 4288, Size: 2048)
 // (op 244) LoadLDSTile{Value: BF8x4}(245) END
 // (op 244) Deallocate{}(1224) BEGIN
 // (op 1224) Deallocate 652
 // Deleting tag 652
 // (op 244) Deallocate{}(1224) END
 // (op 244) end: generate({245})
 // (op 58) SetCoordinate(244) END
v_add_u32 v13, 64, v13 // (op 550) 
 // (op 58) Assign VGPR Add(DataFlowTag(522)U32, ShiftL(Convert(DataFlowTag(523)U64)U32, 1:U32)U32)U32(550) END
 // (op 58) Deallocate{}(1220) BEGIN
 // (op 1220) Deallocate 468
 // Deleting tag 468
 // (op 58) Deallocate{}(1220) END
buffer_load_ubyte v12, v51, s[52:55], 0 offen offset:6 // (op 258) Load value
// VMEM: Expected stall of 158. CBNW: 3, Inc: 3
// VMEM: Expected complete at 3356 (current 2876)
 // (op 319) LoadTiled Value: E8M0(258) END
 // (op 319) end: generate({258})
 // (op 58) SetCoordinate(319) END
 // (op 58) Barrier(1152) BEGIN
 // (op 58) Barrier(1222) BEGIN
 // (op 58) Deallocate{}(1258) BEGIN
 // (op 1258) Deallocate 562
 // Freeing Offset562: 1 VGPR (Value: UInt32) (op 25): v51

 // Deleting tag 562
 // (op 1258) Deallocate 564
 // Deleting tag 564
 // (op 58) Deallocate{}(1258) END
s_barrier  // (op 1152) 
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 3200, Size: 1024)
 // (op 58) Barrier(1152) END
 // (op 58) SetCoordinate(304) BEGIN
 // (op 304) generate({})
 // (op 304) end: generate({})
 // (op 304) generate({242})
 // (op 304) LoadTiled Value: BF8x4(242) BEGIN
 // (op 242) GEN: loadMacroTileVGPRCI Value: BF8x4
 // (op 242) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 242) Generate Convert(Tensor_0_stride_0_0:I64)U64 into Stride506: VGPR Value: UInt64 x 1: (unallocated)
 // (op 242) reg expression
 // (op 58) Assign VGPR Add(DataFlowTag(560)U32, ShiftL(Convert(DataFlowTag(561)U64)U32, 1:U32)U32)U32(678) BEGIN
 // (op 678) Assign dim(560) = Add(DataFlowTag(560)U32, ShiftL(Convert(DataFlowTag(561)U64)U32, 1:U32)U32)U32
 // (op 678) Generate Add(DataFlowTag(560)U32, ShiftL(Convert(DataFlowTag(561)U64)U32, 1:U32)U32)U32 into DataFlowTag560: VGPR Value: UInt32 x 1: v21
 // (op 678) reg expression
 // (op 678) Add({DataFlowTag560: v21:U32}, 4:U32)U32
 // (op 58) SetCoordinate(246) BEGIN
 // (op 246) generate({})
 // (op 246) end: generate({})
 // (op 246) generate({247})
 // (op 246) LoadLDSTile{Value: FP4x8}(247) BEGIN
 // (op 247) GEN: loadMacroTileWAVELDS OP 247 LDS 472 WaveTile 169
 // (op 247) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 247) Generate 0:U32 into nullptr
 // tag 484: v**UNALLOCATED**
 // (op 247) 	Dir = Load
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag658: VGPR Value: UInt32 x 1: v8
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
 // 	info.offset = Literal Value: UInt32 x 0: 3200
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

s_barrier  // (op 1222) 
 // (op 58) Barrier(1222) END
v_add_u32 v21, 4, v21 // (op 678) 
 // (op 58) Assign VGPR Add(DataFlowTag(560)U32, ShiftL(Convert(DataFlowTag(561)U64)U32, 1:U32)U32)U32(678) END
// Allocated Stride506: 2 VGPRs (Value: UInt64) (op 242): v66, v67
v_mov_b64 v[66:67], s[2:3] // (op 242) convert
 // (op 242) Generate 1:U64 into nullptr
 // (op 242) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[40:47]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride506Tensor_0_stride_0_0: VGPR Value: UInt64 x 1: v[66:67]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(16:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(16:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 1 VGPR (Value: UInt32) (op 242): v16
v_mov_b32 v16, v64 // (op 242) 
 // (op 242)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride506Tensor_0_stride_0_0: v[66:67]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
buffer_load_dwordx4 v[40:43], v16, s[40:43], 0 offen offset:192 // (op 242) Load value
// VMEM: Expected complete at 3362 (current 2882)
 // (op 242) Generate Add(v16:U32, v66:R)U32 into VGPR Value: UInt32 x 1: v16
 // (op 242) reg expression
 // (op 242) reg expression
 // (op 242) Add(v16:U32, v66:R)U32
v_add_u32 v16, v16, v66 // (op 242) 
// Allocated : 4 VGPRs (Value: FP4x8) (op 247): v68, v69, v70, v71
ds_read_b128 v[68:71], v8 offset:3200 // (op 247) Load local data 
// Modelled addresses (normalized): [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 16, 48, 80, 112, 144, 176, 208, 240, 272, 304, 336, 368, 400, 432, 464, 496, 528, 560, 592, 624, 656, 688, 720, 752, 784, 816, 848, 880, 912, 944, 976, 1008]
// DSMEM: Expected complete at 3205 (current 3110)
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 3200, Size: 1024)
 // (op 246) LoadLDSTile{Value: FP4x8}(247) END
 // (op 246) Deallocate{}(1230) BEGIN
 // (op 1230) Deallocate 660
 // Deleting tag 660
 // (op 246) Deallocate{}(1230) END
 // (op 246) end: generate({247})
 // (op 58) SetCoordinate(246) END
buffer_load_dwordx4 v[44:47], v16, s[40:43], 0 offen offset:192 // (op 242) Load value
// VMEM: Expected stall of 310. CBNW: 5, Inc: 3
// VMEM: Expected complete at 3675 (current 3195)
 // Freeing Stride506: 2 VGPRs (Value: UInt64) (op 242): v66, v67

 // Freeing : 1 VGPR (Value: UInt32) (op 242): v16

 // (op 304) LoadTiled Value: BF8x4(242) END
 // (op 304) end: generate({242})
 // (op 58) SetCoordinate(304) END
 // (op 58) Deallocate{}(1226) BEGIN
 // (op 1226) Deallocate 472
 // Deleting tag 472
 // (op 58) Deallocate{}(1226) END
 // (op 58) Deallocate{}(1252) BEGIN
 // (op 1252) Deallocate 505
 // Freeing Offset505: 1 VGPR (Value: UInt32) (op 4): v64

 // Deleting tag 505
 // (op 1252) Deallocate 507
 // Deleting tag 507
 // (op 58) Deallocate{}(1252) END
 // (op 58) Barrier(1158) BEGIN
 // (op 58) Barrier(1228) BEGIN
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
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = Offset543: VGPR Value: UInt32 x 1: v62
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

 // (op 58) Assign VGPR Add(DataFlowTag(503)U32, ShiftL(Convert(DataFlowTag(504)U64)U32, 1:U32)U32)U32(486) BEGIN
 // (op 486) Assign dim(503) = Add(DataFlowTag(503)U32, ShiftL(Convert(DataFlowTag(504)U64)U32, 1:U32)U32)U32
 // (op 486) Generate Add(DataFlowTag(503)U32, ShiftL(Convert(DataFlowTag(504)U64)U32, 1:U32)U32)U32 into DataFlowTag503: VGPR Value: UInt32 x 1: v15
 // (op 486) reg expression
 // (op 486) Add({DataFlowTag503: v15:U32}, 128:U32)U32
s_barrier  // (op 1228) 
 // (op 58) Barrier(1228) END
v_add_u32 v15, 128, v15 // (op 486) 
 // (op 58) Assign VGPR Add(DataFlowTag(503)U32, ShiftL(Convert(DataFlowTag(504)U64)U32, 1:U32)U32)U32(486) END
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1158) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 6336, Size: 64)
 // (op 58) Barrier(1158) END
buffer_load_ubyte v14, v62, s[48:51], 0 offen offset:6 // (op 257) Load value
// VMEM: Expected stall of 157. CBNW: 6, Inc: 3
// VMEM: Expected complete at 3837 (current 3357)
 // (op 314) LoadTiled Value: E8M0(257) END
 // (op 314) end: generate({257})
 // (op 58) SetCoordinate(314) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 6336
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
 // 	info.rowOffsetReg = DataFlowTag666: VGPR Value: UInt32 x 1: v5
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
 // 	info.offset = Literal Value: UInt32 x 0: 6336
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

 // (op 58) Deallocate{}(1256) BEGIN
 // (op 1256) Deallocate 543
 // Freeing Offset543: 1 VGPR (Value: UInt32) (op 17): v62

 // Deleting tag 543
 // (op 1256) Deallocate 545
 // Deleting tag 545
 // (op 58) Deallocate{}(1256) END
// Allocated : 1 VGPR (Value: E8M0) (op 249): v16
ds_read_u8 v16, v5 offset:6336 // (op 249) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 3679 (current 3584)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 6336, Size: 64)
 // (op 248) LoadLDSTile{Value: E8M0}(249) END
 // (op 248) Deallocate{}(1236) BEGIN
 // (op 1236) Deallocate 668
 // Deleting tag 668
 // (op 248) Deallocate{}(1236) END
 // (op 248) end: generate({249})
 // (op 58) SetCoordinate(248) END
 // (op 58) Assign VGPR Add(DataFlowTag(541)U32, ShiftL(Convert(DataFlowTag(542)U64)U32, 1:U32)U32)U32(614) BEGIN
 // (op 614) Assign dim(541) = Add(DataFlowTag(541)U32, ShiftL(Convert(DataFlowTag(542)U64)U32, 1:U32)U32)U32
 // (op 614) Generate Add(DataFlowTag(541)U32, ShiftL(Convert(DataFlowTag(542)U64)U32, 1:U32)U32)U32 into DataFlowTag541: VGPR Value: UInt32 x 1: v17
 // (op 614) reg expression
 // (op 614) Add({DataFlowTag541: v17:U32}, 4:U32)U32
 // (op 58) Barrier(1107) BEGIN
 // (op 58) Deallocate{}(1232) BEGIN
 // (op 1232) Deallocate 476
 // Deleting tag 476
 // (op 58) Deallocate{}(1232) END
s_barrier  // (op 1107) 
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 0, Size: 1024)
 // (op 58) Barrier(1107) END
 // (op 58) Barrier(1164) BEGIN
 // (op 58) Barrier(1234) BEGIN
 // (op 58) StoreLDSTile Value: FP4x8(13) BEGIN
 // (op 13) GEN: StoreLDSTile
 // (op 13) GEN: storeMacroTileLDS OP 13 LDS 25 MacroTile 9
 // (op 13) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 13) Generate 0:U32 into nullptr
 // (op 13) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[22:25]
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
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[22:25]
 // 	info.rowOffsetReg = DataFlowTag588: VGPR Value: UInt32 x 1: v20
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
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_barrier  // (op 1164) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 4224, Size: 64)
 // (op 58) Barrier(1164) END
s_barrier  // (op 1234) 
 // (op 58) Barrier(1234) END
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

 // (op 251) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = DataFlowTag674: VGPR Value: UInt32 x 1: v10
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
 // (op 251) 	info.m = 1
 // 	info.n = 1
 // 	elementsPerMove = 1
 // 	bytesPerMove = 1
 // 	rowStride = 0
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 8
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

v_add_u32 v17, 4, v17 // (op 614) 
 // (op 58) Assign VGPR Add(DataFlowTag(541)U32, ShiftL(Convert(DataFlowTag(542)U64)U32, 1:U32)U32)U32(614) END
// Allocated : 1 VGPR (Value: E8M0) (op 251): v51
ds_read_u8 v51, v10 offset:4224 // (op 251) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 3684 (current 3589)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 4224, Size: 64)
 // (op 250) LoadLDSTile{Value: E8M0}(251) END
 // (op 250) Deallocate{}(1242) BEGIN
 // (op 1242) Deallocate 676
 // Deleting tag 676
 // (op 250) Deallocate{}(1242) END
 // (op 250) end: generate({251})
 // (op 58) SetCoordinate(250) END
s_waitcnt vmcnt(9)
ds_write_b128 v20, v[22:25] // (op 13) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008]
// DSMEM: Expected complete at 3686 (current 3591)
// Extra dsts: LDS Value: FP4x8 x 0: LDS:(Offset: 0, Size: 1024)
 // (op 58) StoreLDSTile Value: FP4x8(13) END
 // (op 58) Deallocate{}(1238) BEGIN
 // (op 1238) Deallocate 480
 // Deleting tag 480
 // (op 58) Deallocate{}(1238) END
 // (op 58) Deallocate{}(1172) BEGIN
 // (op 1172) Deallocate 590
 // Deleting tag 590
 // (op 58) Deallocate{}(1172) END
 // (op 58) Multiply(252) BEGIN
 // (op 252) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile8xFP4, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)S into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
 // (op 58) Barrier(1240) BEGIN
 // (op 58) Barrier(1135) BEGIN
s_barrier  // (op 1240) 
 // (op 58) Barrier(1240) END
s_barrier  // (op 1135) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 1024, Size: 64)
 // (op 58) Barrier(1135) END
s_waitcnt lgkmcnt(1)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[54:61], v[68:71], a[0:15], v16, v51 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 252) 
 // (op 58) Multiply(252) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 1024
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
 // 	info.rowOffsetReg = DataFlowTag620: VGPR Value: UInt32 x 1: v49
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
 // 	info.offset = Literal Value: UInt32 x 0: 1024
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

 // (op 58) Deallocate{}(1244) BEGIN
 // (op 1244) Deallocate 486
 // Freeing : 1 VGPR (Value: E8M0) (op 249): v16

 // Deleting tag 486
 // (op 1244) Deallocate 488
 // Freeing : 1 VGPR (Value: E8M0) (op 251): v51

 // Deleting tag 488
 // (op 1244) Deallocate 484
 // Freeing : 4 VGPRs (Value: FP4x8) (op 247): v68, v69, v70, v71

 // Deleting tag 484
 // (op 1244) Deallocate 482
 // Freeing : 8 VGPRs (Value: BF8x4) (op 245): v54, v55, v56, v57, v58, v59, v60, v61

 // Deleting tag 482
 // (op 58) Deallocate{}(1244) END
s_waitcnt vmcnt(8)
ds_write_b8 v49, v18 offset:1024 // (op 30) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 3783 (current 3688)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 1024, Size: 64)
 // (op 58) StoreLDSTile Value: E8M0(30) END
 // (op 58) Deallocate{}(1176) BEGIN
 // (op 1176) Deallocate 622
 // Deleting tag 622
 // (op 58) Deallocate{}(1176) END
 // (op 58) Barrier(1094) BEGIN
s_barrier  // (op 1094) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
 // (op 58) Barrier(1094) END
 // (op 58) StoreLDSTile Value: BF8x4(6) BEGIN
 // (op 6) GEN: StoreLDSTile
 // (op 6) GEN: storeMacroTileLDS OP 6 LDS 17 MacroTile 8
 // (op 6) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 6) Generate 64:U32 into nullptr
 // (op 6) Generate 1:U32 into nullptr
 // (op 6) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[28:35]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 64
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1088
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
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[28:35]
 // 	info.rowOffsetReg = DataFlowTag572: VGPR Value: UInt32 x 1: v26
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 64
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1088
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 6) 	info.m = 2
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 64
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(7)
ds_write_b128 v26, v[28:31] offset:1088 // (op 6) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 128, 144, 160, 176, 256, 272, 288, 304, 384, 400, 416, 432, 512, 528, 544, 560, 640, 656, 672, 688, 768, 784, 800, 816, 896, 912, 928, 944, 1024, 1040, 1056, 1072, 1152, 1168, 1184, 1200, 1280, 1296, 1312, 1328, 1408, 1424, 1440, 1456, 1536, 1552, 1568, 1584, 1664, 1680, 1696, 1712, 1792, 1808, 1824, 1840, 1920, 1936, 1952, 1968]
// DSMEM: Expected complete at 3786 (current 3691)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
s_waitcnt vmcnt(6)
ds_write_b128 v26, v[32:35] offset:1152 // (op 6) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 128, 144, 160, 176, 256, 272, 288, 304, 384, 400, 416, 432, 512, 528, 544, 560, 640, 656, 672, 688, 768, 784, 800, 816, 896, 912, 928, 944, 1024, 1040, 1056, 1072, 1152, 1168, 1184, 1200, 1280, 1296, 1312, 1328, 1408, 1424, 1440, 1456, 1536, 1552, 1568, 1584, 1664, 1680, 1696, 1712, 1792, 1808, 1824, 1840, 1920, 1936, 1952, 1968]
// DSMEM: Expected complete at 3788 (current 3693)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
 // (op 58) StoreLDSTile Value: BF8x4(6) END
 // (op 58) Deallocate{}(1170) BEGIN
 // (op 1170) Deallocate 574
 // Deleting tag 574
 // (op 58) Deallocate{}(1170) END
 // (op 58) Barrier(1121) BEGIN
s_barrier  // (op 1121) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 3136, Size: 64)
 // (op 58) Barrier(1121) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 3136
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
 // 	info.rowOffsetReg = DataFlowTag604: VGPR Value: UInt32 x 1: v48
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
 // 	info.offset = Literal Value: UInt32 x 0: 3136
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
ds_write_b8 v48, v19 offset:3136 // (op 21) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 89. CBNW: 1, Inc: 3
// DSMEM: Expected complete at 3880 (current 3785)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 3136, Size: 64)
 // (op 58) StoreLDSTile Value: E8M0(21) END
 // (op 58) Deallocate{}(1174) BEGIN
 // (op 1174) Deallocate 606
 // Deleting tag 606
 // (op 58) Deallocate{}(1174) END
 // (op 58) NOP(407) BEGIN
 // (op 407) generate({})
 // (op 407) end: generate({})
 // (op 58) NOP(407) END
 // (op 58) end: generate({60, 288, 1091})
 // (op 58) For Loop Increment
 // (op 58) generate({})
 // (op 58) end: generate({})
 // (op 58) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 if true)
 // (op 58) Generate LessThan(DataFlowTag(395)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL into SCC Value: Bool x 1: scc
 // (op 58) LessThan({DataFlowTag395: s0:U32}, BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32)BL
 // (op 58) BitwiseAnd(Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32, 4294967294:U32)U32
 // (op 58) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 58): s14, s15
s_ashr_i64 s[14:15], s[28:29], 6 // (op 58) 
// Allocated : 1 SGPR (Value: UInt32) (op 58): s1
s_mov_b32 s1, s14 // (op 58) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 58): s14, s15

// Allocated : 1 SGPR (Value: UInt32) (op 58): s14
s_and_b32 s14, s1, 4294967294 // (op 58) 
 // Freeing : 1 SGPR (Value: UInt32) (op 58): s1

s_cmp_lt_u32 s0, s14 // (op 58) 
 // Freeing : 1 SGPR (Value: UInt32) (op 58): s14

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 58) 
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 // (op 58) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoop_58 if true)
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoop_58:
 // (op 58) 
 // (op 58) Unlock For Loop
 // Unlock None
 // (op 467) ForLoopOp KLoop: LessThan(DataFlowTag(395)U32, BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32)BL(58) END
 // (op 467) Deallocate{}(1180) BEGIN
 // (op 1180) Deallocate 508
 // Deleting tag 508
 // (op 1180) Deallocate 10
 // Freeing : 1 VGPR (Value: E8M0) (op 375): v19

 // Deleting tag 10
 // (op 1180) Deallocate 671
 // Deleting tag 671
 // (op 1180) Deallocate 646
 // Freeing DataFlowTag646: 1 VGPR (Value: UInt32) (op 970): v50

 // Deleting tag 646
 // (op 1180) Deallocate 647
 // Deleting tag 647
 // (op 1180) Deallocate 654
 // Freeing DataFlowTag654: 1 VGPR (Value: UInt32) (op 994): v27

 // Deleting tag 654
 // (op 1180) Deallocate 657
 // Deleting tag 657
 // (op 1180) Deallocate 663
 // Deleting tag 663
 // (op 1180) Deallocate 665
 // Deleting tag 665
 // (op 1180) Deallocate 649
 // Deleting tag 649
 // (op 1180) Deallocate 395
 // Freeing DataFlowTag395: 1 SGPR (Value: UInt32) (op 59): s0

 // Deleting tag 395
 // (op 1180) Deallocate 670
 // Freeing DataFlowTag670: 1 VGPR (Value: UInt32) (op 1042): v53

 // Deleting tag 670
 // (op 1180) Deallocate 673
 // Deleting tag 673
 // (op 1180) Deallocate 662
 // Freeing DataFlowTag662: 1 VGPR (Value: UInt32) (op 1018): v52

 // Deleting tag 662
 // (op 1180) Deallocate 655
 // Deleting tag 655
 // (op 1180) Deallocate 506
 // Deleting tag 506
 // (op 1180) Deallocate 504
 // Deleting tag 504
 // (op 1180) Deallocate 589
 // Deleting tag 589
 // (op 1180) Deallocate 503
 // Freeing DataFlowTag503: 1 VGPR (Value: UInt32) (op 494): v15

 // Deleting tag 503
 // (op 1180) Deallocate 620
 // Freeing DataFlowTag620: 1 VGPR (Value: UInt32) (op 883): v49

 // Deleting tag 620
 // (op 1180) Deallocate 470
 // Deleting tag 470
 // (op 1180) Deallocate 478
 // Deleting tag 478
 // (op 1180) Deallocate 466
 // Deleting tag 466
 // (op 1180) Deallocate 605
 // Deleting tag 605
 // (op 1180) Deallocate 604
 // Freeing DataFlowTag604: 1 VGPR (Value: UInt32) (op 830): v48

 // Deleting tag 604
 // (op 1180) Deallocate 572
 // Freeing DataFlowTag572: 1 VGPR (Value: UInt32) (op 724): v26

 // Deleting tag 572
 // (op 1180) Deallocate 573
 // Deleting tag 573
 // (op 1180) Deallocate 575
 // Deleting tag 575
 // (op 1180) Deallocate 588
 // Freeing DataFlowTag588: 1 VGPR (Value: UInt32) (op 777): v20

 // Deleting tag 588
 // (op 1180) Deallocate 591
 // Deleting tag 591
 // (op 1180) Deallocate 607
 // Deleting tag 607
 // (op 1180) Deallocate 621
 // Deleting tag 621
 // (op 1180) Deallocate 623
 // Deleting tag 623
 // (op 1180) Deallocate 8
 // Freeing : 8 VGPRs (Value: BF8x4) (op 372): v28, v29, v30, v31, v32, v33, v34, v35

 // Deleting tag 8
 // (op 1180) Deallocate 522
 // Freeing DataFlowTag522: 1 VGPR (Value: UInt32) (op 558): v13

 // Deleting tag 522
 // (op 1180) Deallocate 565
 // Deleting tag 565
 // (op 1180) Deallocate 560
 // Freeing DataFlowTag560: 1 VGPR (Value: UInt32) (op 686): v21

 // Deleting tag 560
 // (op 1180) Deallocate 546
 // Deleting tag 546
 // (op 1180) Deallocate 544
 // Deleting tag 544
 // (op 1180) Deallocate 541
 // Freeing DataFlowTag541: 1 VGPR (Value: UInt32) (op 622): v17

 // Deleting tag 541
 // (op 1180) Deallocate 9
 // Freeing : 4 VGPRs (Value: FP4x8) (op 366): v22, v23, v24, v25

 // Deleting tag 9
 // (op 1180) Deallocate 523
 // Deleting tag 523
 // (op 1180) Deallocate 474
 // Deleting tag 474
 // (op 1180) Deallocate 542
 // Deleting tag 542
 // (op 1180) Deallocate 561
 // Deleting tag 561
 // (op 1180) Deallocate 525
 // Deleting tag 525
 // (op 1180) Deallocate 11
 // Freeing : 1 VGPR (Value: E8M0) (op 369): v18

 // Deleting tag 11
 // (op 1180) Deallocate 563
 // Deleting tag 563
 // (op 1180) Deallocate 527
 // Deleting tag 527
 // (op 467) Deallocate{}(1180) END
 // (op 467) end: generate({462})
 // (op 467) Unlock Scope 467
 // Unlock None
 // (op 340) Scope(467) END
 // (op 340) SetCoordinate(217) BEGIN
 // (op 217) generate({})
 // (op 217) end: generate({})
 // (op 217) generate({218})
 // (op 217) SetCoordinate(218) BEGIN
 // (op 218) generate({})
 // (op 218) end: generate({})
 // (op 218) generate({444})
 // (op 218) Scope(444) BEGIN
 // (op 444) Lock Scope 444
 // Lock Branch
 // (op 444) generate({438})
 // (op 444) NOP(438) BEGIN
 // (op 438) generate({})
 // (op 438) end: generate({})
 // (op 444) NOP(438) END
 // (op 444) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, {Tile: ShiftL(LogicalShiftR(DataFlowTag(76)U32, 2:I)U32, 1:U32)U32})U32}, Convert(Tensor_0_stride_0_0:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32}(449) BEGIN
 // (op 449) Assign dim(490) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, {Tile: ShiftL(LogicalShiftR(DataFlowTag(76)U32, 2:I)U32, 1:U32)U32})U32}, Convert(Tensor_0_stride_0_0:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32}
 // tag 490: v**UNALLOCATED**
 // (op 449) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, {Tile: ShiftL(LogicalShiftR(DataFlowTag(76)U32, 2:I)U32, 1:U32)U32})U32}, Convert(Tensor_0_stride_0_0:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32} into DataFlowTag490: VGPR Value: UInt32 x 1: (unallocated)
 // (op 449) reg expression
 // (op 449) reg expression
 // (op 449) reg expression
 // (op 449) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
 // (op 449) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 449): v13
v_lshrrev_b32 v13, 2, v1 // (op 449) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 449): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 449) 
 // (op 449) BEGIN: Tile
 // (op 449) {Tile: ShiftL(v13:U32, 1:U32)U32}
 // (op 449) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 449)         value (VGPR Value: UInt32 x 1: v13) 
 // (op 449)         shiftAmount (Literal Value: UInt32 x 0: 1)
 // (op 449) convert
// Allocated : 1 VGPR (Value: UInt32) (op 449): v15
v_lshlrev_b32 v15, 1, v13 // (op 449) 
 // (op 449) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 449): v13

 // (op 449) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 449) BEGIN: Tile
// Allocated : 1 SGPR (Value: UInt32) (op 449): s14
s_lshr_b32 s14, s0, 1 // (op 449) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 449): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 449): v13
v_lshl_add_u32 v13, s16, 5, v15 // (op 449) 
 // (op 449) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 449): v15

 // (op 449) reg expression
 // (op 449) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
 // (op 449) convert
// Allocated : 1 VGPR (Value: UInt32) (op 449): v15
v_and_b32 v15, -4, v1 // (op 449) 
 // (op 449) BEGIN: Flatten[1]
 // (op 449) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v15:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 449): v16
v_sub_u32 v16, v1, v15 // (op 449) 
 // (op 449) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 449): v15

 // (op 449) BEGIN: Tile
 // (op 449) {Tile: ShiftL({Flatten[1]: v16:U32}, 4:U32)U32}
 // (op 449) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 449)         value (Flatten[1]: VGPR Value: UInt32 x 1: v16) 
 // (op 449)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 449): v15
v_lshlrev_b32 v15, 4, v16 // (op 449) 
 // (op 449) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 449): v16

 // (op 449) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 449): v16
v_lshl_add_u32 v16, s14, 7, v15 // (op 449) 
 // (op 449) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 449): v15

 // Freeing : 1 SGPR (Value: UInt32) (op 449): s14

 // (op 449) BEGIN: Split
// Allocated DataFlowTag490: 1 VGPR (Value: UInt32) (op 449): v15
v_mul_lo_u32 v15, v13, s2 // (op 449) 
v_add_u32 v15, v15, v16 // (op 449) 
 // (op 449) END: Split
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 449): v16

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 449): v13

 // (op 444) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(427)U32, 5:U32, {Tile: ShiftL(LogicalShiftR(DataFlowTag(76)U32, 2:I)U32, 1:U32)U32})U32}, Convert(Tensor_0_stride_0_0:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(76)U32, BitwiseAnd(DataFlowTag(76)U32, -4:I)U32)U32}, 4:U32)U32})U32})U32}(449) END
 // (op 444) Assign VGPR 64:U64(450) BEGIN
 // (op 450) Assign dim(491) = 64:U64
 // (op 450) Assign stride expression
 // (op 444) Assign VGPR 64:U64(450) END
 // (op 444) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(451) BEGIN
 // (op 451) Assign dim(493) = Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 451) Generate Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag493: SGPR Buffer: None x 1: s[40:43]
 // (op 451) reg expression
 // (op 451) reg expression
 // (op 451) Multiply({Tensor_0_stride_0_0: s[2:3]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 451) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 451)           lhs (Tensor_0_stride_0_0: SGPR Value: Int64 x 1: s[2:3]) 
 // (op 451)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 451): s0, s1
s_mul_i32 s1, s2, s7 // (op 451) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 451): s14
s_mul_i32 s14, s3, s6 // (op 451) most significant: low of high * low
s_mul_i32 s0, s2, s6 // (op 451) least significant
// Allocated : 1 SGPR (Value: Int32) (op 451): s15
s_mul_hi_u32 s15, s2, s6 // (op 451) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 451) most significant: sum
s_add_u32 s1, s1, s15 // (op 451) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 451): s15

 // Freeing : 1 SGPR (Value: Int32) (op 451): s14

 // (op 451) reg expression
 // (op 451) reg expression
s_mov_b32 s40, s4 // (op 451) reg expression
s_mov_b32 s41, s5 // (op 451) reg expression
s_mov_b32 s42, s0 // (op 451) reg expression
s_mov_b32 s43, 131072 // (op 451) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 451): s0, s1

 // (op 444) Assign SGPR Concatenate(Tensor_0_pointer:PG, BitFieldExtract(Multiply(Tensor_0_stride_0_0:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(451) END
 // (op 444) NOP(440) BEGIN
 // (op 440) generate({})
 // (op 440) end: generate({})
 // (op 444) NOP(440) END
 // (op 444) Assign VGPR Convert(Tensor_0_stride_0_0:I64)U64(456) BEGIN
 // (op 456) Assign dim(495) = Convert(Tensor_0_stride_0_0:I64)U64
 // (op 456) Assign stride expression
 // (op 444) Assign VGPR Convert(Tensor_0_stride_0_0:I64)U64(456) END
 // (op 444) NOP(441) BEGIN
 // (op 441) generate({})
 // (op 441) end: generate({})
 // (op 444) NOP(441) END
 // (op 444) Assign VGPR 1:U64(459) BEGIN
 // (op 459) Assign dim(497) = 1:U64
 // (op 459) Assign stride expression
 // (op 444) Assign VGPR 1:U64(459) END
 // (op 444) NOP(507) BEGIN
 // (op 507) generate({})
 // (op 507) end: generate({})
 // (op 444) NOP(507) END
 // (op 444) Assign VGPR Convert(LogicalShiftR({Split: AddShiftL(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64, 2:U32)U64}, 3:U32)U64)U32(516) BEGIN
 // (op 516) Assign dim(509) = Convert(LogicalShiftR({Split: AddShiftL(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64, 2:U32)U64}, 3:U32)U64)U32
 // tag 509: v**UNALLOCATED**
 // (op 516) Generate Convert(LogicalShiftR({Split: AddShiftL(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64, 2:U32)U64}, 3:U32)U64)U32 into DataFlowTag509: VGPR Value: UInt32 x 1: (unallocated)
 // (op 516) reg expression
 // (op 516) reg expression
 // (op 516) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
 // (op 516) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 516): v13
v_and_b32 v13, -2, v1 // (op 516) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 516): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 516) 
 // (op 516) BEGIN: Flatten[1]
 // (op 516) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v13:U32)U32}
 // (op 516) convert
// Allocated : 1 VGPR (Value: UInt32) (op 516): v16
v_sub_u32 v16, v1, v13 // (op 516) 
 // (op 516) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 516): v13

 // (op 516) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 516) BEGIN: Tile
 // (op 516) {Tile: ShiftL({Flatten[1]: v16:U32}, 5:U32)U32}
 // (op 516) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 516)         value (Flatten[1]: VGPR Value: UInt32 x 1: v16) 
 // (op 516)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 SGPR (Value: UInt32) (op 516): s14
s_lshr_b32 s14, s0, 1 // (op 516) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 516): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 516): v13
v_lshlrev_b32 v13, 5, v16 // (op 516) 
 // (op 516) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 516): v16

 // (op 516) reg expression
 // (op 516) BEGIN: Tile
 // (op 516) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 516): v16
v_lshl_add_u32 v16, s14, 7, v13 // (op 516) 
 // (op 516) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 516): v13

 // Freeing : 1 SGPR (Value: UInt32) (op 516): s14

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 516): v34, v35
v_mov_b32 v35, 0 // (op 516) convert
v_mov_b32 v34, v16 // (op 516) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 516): v16

 // (op 516) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 516): v13
v_lshrrev_b32 v13, 1, v1 // (op 516) 
 // (op 516) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 516): v16
v_lshl_add_u32 v16, s17, 5, v13 // (op 516) 
 // (op 516) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 516): v13

 // (op 516) Multiply({Tile: v16:U32}, {Tensor_2_stride_1_3: s[8:9]:I64})I64
 // (op 516) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 516)           lhs (Tile: VGPR Value: UInt32 x 1: v16) 
 // (op 516)           rhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9])
// Allocated : 2 VGPRs (Value: Int64) (op 516): v32, v33
v_mul_lo_u32 v33, v16, s9 // (op 516) most significant: low of low * high
 // (op 516) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 516): v13
v_mul_hi_u32 v13, v16, s8 // (op 516) most significant: high of low * low
v_mul_lo_u32 v32, v16, s8 // (op 516) least significant: low of low * low
v_add_u32 v33, v33, v13 // (op 516) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 516): v13

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 516): v16

 // (op 516) BEGIN: Split
 // (op 516) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 516)      lhs (Tile: VGPR Value: UInt64 x 1: v[34:35]) 
 // (op 516)      rhs (VGPR Value: Int64 x 1: v[32:33])
// Allocated : 2 VGPRs (Value: UInt64) (op 516): v16, v17
// Allocated : 2 SGPRs (Value: Bool64) (op 516): s0, s1
v_add_co_u32 v16, s[0:1], v34, v32 // (op 516) least significant half
v_addc_co_u32 v17, s[0:1], v35, v33, s[0:1] // (op 516) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 516): s0, s1

 // (op 516) ShiftL: dest (VGPR Value: UInt64 x 1: v[16:17]) = 
 // (op 516)         value (VGPR Value: UInt64 x 1: v[16:17]) 
 // (op 516)         shiftAmount (Literal Value: UInt32 x 0: 2)
v_lshlrev_b64 v[16:17], 2, v[16:17] // (op 516) 
 // (op 516) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 516): v32, v33

 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 516): v34, v35

 // (op 516) LogicalShiftR({Split: v[16:17]:U64}, 3:U32)U64
// Allocated  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 516): v18, v19
v_lshrrev_b64 v[18:19], 3, v[16:17] // (op 516) 
 // Freeing Split: 2 VGPRs (Value: UInt64) (op 516): v16, v17

// Allocated DataFlowTag509: 1 VGPR (Value: UInt32) (op 516): v13
v_mov_b32 v13, v18 // (op 516) convert
 // Freeing  convertForwardSubset: 2 VGPRs (Value: UInt64) (op 516): v18, v19

 // (op 444) Assign VGPR Convert(LogicalShiftR({Split: AddShiftL(Convert({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 7:U32, {Tile: ShiftL({Flatten[1]: Subtract(DataFlowTag(142)U32, BitwiseAnd(DataFlowTag(142)U32, -2:I)U32)U32}, 5:U32)U32})U32})U64, Multiply({Tile: ShiftLAdd(DataFlowTag(429)U32, 5:U32, LogicalShiftR(DataFlowTag(142)U32, 1:I)U32)U32}, Tensor_2_stride_1_3:I64)I64, 2:U32)U64}, 3:U32)U64)U32(516) END
 // (op 444) Assign VGPR 32:U64(517) BEGIN
 // (op 517) Assign dim(510) = 32:U64
 // (op 517) Assign stride expression
 // (op 444) Assign VGPR 32:U64(517) END
 // (op 444) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(518) BEGIN
 // (op 518) Assign dim(512) = Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 518) Generate Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag512: SGPR Buffer: None x 1: s[44:47]
 // (op 518) reg expression
 // (op 518) reg expression
 // (op 518) Multiply({Tensor_2_stride_1_3: s[8:9]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 518) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 518)           lhs (Tensor_2_stride_1_3: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 518)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 518): s0, s1
s_mul_i32 s1, s8, s13 // (op 518) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 518): s14
s_mul_i32 s14, s9, s12 // (op 518) most significant: low of high * low
s_mul_i32 s0, s8, s12 // (op 518) least significant
// Allocated : 1 SGPR (Value: Int32) (op 518): s15
s_mul_hi_u32 s15, s8, s12 // (op 518) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 518) most significant: sum
s_add_u32 s1, s1, s15 // (op 518) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 518): s15

 // Freeing : 1 SGPR (Value: Int32) (op 518): s14

 // (op 518) ShiftL(s[0:1]:I64, 2:U32)I64
 // (op 518) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 518)         value (SGPR Value: Int64 x 1: s[0:1]) 
 // (op 518)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 518): s14, s15
s_lshl_b64 s[14:15], s[0:1], 2 // (op 518) 
 // Freeing : 2 SGPRs (Value: Int64) (op 518): s0, s1

 // (op 518) ArithmeticShiftR(s[14:15]:I64, 3:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 518): s0, s1
s_ashr_i64 s[0:1], s[14:15], 3 // (op 518) 
 // Freeing : 2 SGPRs (Value: Int64) (op 518): s14, s15

 // (op 518) reg expression
 // (op 518) reg expression
s_mov_b32 s44, s10 // (op 518) reg expression
s_mov_b32 s45, s11 // (op 518) reg expression
s_mov_b32 s46, s0 // (op 518) reg expression
s_mov_b32 s47, 131072 // (op 518) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 518): s0, s1

 // (op 444) Assign SGPR Concatenate(Tensor_2_pointer:PG, BitFieldExtract(ArithmeticShiftR(ShiftL(Multiply(Tensor_2_stride_1_3:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, 3:U32)I64, width:32, offset:0)R, 131072:R)PB(518) END
 // (op 444) NOP(509) BEGIN
 // (op 509) generate({})
 // (op 509) end: generate({})
 // (op 444) NOP(509) END
 // (op 444) Assign VGPR LogicalShiftR(ShiftL(Convert(Tensor_2_stride_1_3:I64)U64, 2:U32)U64, 3:U32)U64(523) BEGIN
 // (op 523) Assign dim(514) = LogicalShiftR(ShiftL(Convert(Tensor_2_stride_1_3:I64)U64, 2:U32)U64, 3:U32)U64
 // (op 523) Assign stride expression
 // (op 444) Assign VGPR LogicalShiftR(ShiftL(Convert(Tensor_2_stride_1_3:I64)U64, 2:U32)U64, 3:U32)U64(523) END
 // (op 444) NOP(510) BEGIN
 // (op 510) generate({})
 // (op 510) end: generate({})
 // (op 444) NOP(510) END
 // (op 444) Assign VGPR 0:U64(526) BEGIN
 // (op 526) Assign dim(516) = 0:U64
 // (op 526) Assign stride expression
 // (op 444) Assign VGPR 0:U64(526) END
 // (op 444) NOP(571) BEGIN
 // (op 571) generate({})
 // (op 571) end: generate({})
 // (op 444) NOP(571) END
 // (op 444) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32})U32}(580) BEGIN
 // (op 580) Assign dim(528) = {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32})U32}
 // tag 528: v**UNALLOCATED**
 // (op 580) Generate {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32})U32} into DataFlowTag528: VGPR Value: UInt32 x 1: (unallocated)
 // (op 580) reg expression
 // (op 580) reg expression
 // (op 580) reg expression
 // (op 580) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
 // (op 580) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 1 VGPR (Value: UInt32) (op 580): v16
v_lshrrev_b32 v16, 1, v1 // (op 580) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 580): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 580) 
 // (op 580) BEGIN: Tile
 // (op 580) convert
// Allocated : 1 VGPR (Value: UInt32) (op 580): v17
v_lshl_add_u32 v17, s16, 5, v16 // (op 580) 
 // (op 580) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 580): v16

 // (op 580) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 580) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 580): s14
s_lshr_b32 s14, s0, 1 // (op 580) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 580): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 580): v16
v_and_b32 v16, -2, v1 // (op 580) 
 // (op 580) reg expression
 // (op 580) BEGIN: Flatten[1]Tile
 // (op 580) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
 // (op 580) convert
// Allocated : 1 VGPR (Value: UInt32) (op 580): v18
v_sub_u32 v18, v1, v16 // (op 580) 
 // (op 580) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 580): v16

 // (op 580) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 580): v16
v_lshl_add_u32 v16, s14, 2, v18 // (op 580) 
 // (op 580) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 580): v18

 // Freeing : 1 SGPR (Value: UInt32) (op 580): s14

 // (op 580) BEGIN: Split
// Allocated DataFlowTag528: 1 VGPR (Value: UInt32) (op 580): v18
v_mul_lo_u32 v18, v17, s20 // (op 580) 
v_add_u32 v18, v18, v16 // (op 580) 
 // (op 580) END: Split
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 580): v16

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 580): v17

 // (op 444) Assign VGPR {Split: MultiplyAdd({Tile: ShiftLAdd(DataFlowTag(431)U32, 5:U32, LogicalShiftR(DataFlowTag(208)U32, 1:I)U32)U32}, Convert(Tensor_4_stride_0_6:I64)U32, {Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(208)U32, BitwiseAnd(DataFlowTag(208)U32, -2:I)U32)U32})U32})U32}(580) END
 // (op 444) Assign VGPR 2:U64(581) BEGIN
 // (op 581) Assign dim(529) = 2:U64
 // (op 581) Assign stride expression
 // (op 444) Assign VGPR 2:U64(581) END
 // (op 444) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(582) BEGIN
 // (op 582) Assign dim(531) = Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 582) Generate Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag531: SGPR Buffer: None x 1: s[48:51]
 // (op 582) reg expression
 // (op 582) reg expression
 // (op 582) Multiply({Tensor_4_stride_0_6: s[20:21]:I64}, {Tensor_0_size_0_1: s[6:7]:I64})I64
 // (op 582) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 582)           lhs (Tensor_4_stride_0_6: SGPR Value: Int64 x 1: s[20:21]) 
 // (op 582)           rhs (Tensor_0_size_0_1: SGPR Value: Int64 x 1: s[6:7])
// Allocated : 2 SGPRs (Value: Int64) (op 582): s0, s1
s_mul_i32 s1, s20, s7 // (op 582) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 582): s14
s_mul_i32 s14, s21, s6 // (op 582) most significant: low of high * low
s_mul_i32 s0, s20, s6 // (op 582) least significant
// Allocated : 1 SGPR (Value: Int32) (op 582): s15
s_mul_hi_u32 s15, s20, s6 // (op 582) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 582) most significant: sum
s_add_u32 s1, s1, s15 // (op 582) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 582): s15

 // Freeing : 1 SGPR (Value: Int32) (op 582): s14

 // (op 582) reg expression
 // (op 582) reg expression
s_mov_b32 s48, s22 // (op 582) reg expression
s_mov_b32 s49, s23 // (op 582) reg expression
s_mov_b32 s50, s0 // (op 582) reg expression
s_mov_b32 s51, 131072 // (op 582) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 582): s0, s1

 // (op 444) Assign SGPR Concatenate(Tensor_4_pointer:PG, BitFieldExtract(Multiply(Tensor_4_stride_0_6:I64, Tensor_0_size_0_1:I64)I64, width:32, offset:0)R, 131072:R)PB(582) END
 // (op 444) NOP(573) BEGIN
 // (op 573) generate({})
 // (op 573) end: generate({})
 // (op 444) NOP(573) END
 // (op 444) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(587) BEGIN
 // (op 587) Assign dim(533) = Convert(Tensor_4_stride_0_6:I64)U64
 // (op 587) Assign stride expression
 // (op 444) Assign VGPR Convert(Tensor_4_stride_0_6:I64)U64(587) END
 // (op 444) NOP(574) BEGIN
 // (op 574) generate({})
 // (op 574) end: generate({})
 // (op 444) NOP(574) END
 // (op 444) Assign VGPR 1:U64(590) BEGIN
 // (op 590) Assign dim(535) = 1:U64
 // (op 590) Assign stride expression
 // (op 444) Assign VGPR 1:U64(590) END
 // (op 444) NOP(635) BEGIN
 // (op 635) generate({})
 // (op 635) end: generate({})
 // (op 444) NOP(635) END
 // (op 444) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(644) BEGIN
 // (op 644) Assign dim(547) = {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}
 // tag 547: v**UNALLOCATED**
 // (op 644) Generate {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32} into DataFlowTag547: VGPR Value: UInt32 x 1: (unallocated)
 // (op 644) reg expression
 // (op 644) reg expression
 // (op 644) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
 // (op 644) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 644): v16
v_and_b32 v16, -2, v1 // (op 644) 
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 644): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 644) 
 // (op 644) BEGIN: Flatten[1]Tile
 // (op 644) {Flatten[1]Tile: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
 // (op 644) convert
// Allocated : 1 VGPR (Value: UInt32) (op 644): v17
v_sub_u32 v17, v1, v16 // (op 644) 
 // (op 644) END: Flatten[1]Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 644): v16

 // (op 644) LogicalShiftR({ convertForwardSubset: s0:U32}, 1:U32)U32
 // (op 644) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 SGPR (Value: UInt32) (op 644): s14
s_lshr_b32 s14, s0, 1 // (op 644) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 644): s0, s1

// Allocated : 1 VGPR (Value: UInt32) (op 644): v16
v_lshrrev_b32 v16, 1, v1 // (op 644) 
 // (op 644) reg expression
 // (op 644) BEGIN: Tile
 // (op 644) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 644): v19
v_lshl_add_u32 v19, s14, 2, v17 // (op 644) 
 // (op 644) END: Tile
 // Freeing Flatten[1]Tile: 1 VGPR (Value: UInt32) (op 644): v17

 // Freeing : 1 SGPR (Value: UInt32) (op 644): s14

 // (op 644) convert
 // (op 644) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 644): v17
v_lshl_add_u32 v17, s17, 5, v16 // (op 644) 
 // (op 644) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 644): v16

 // (op 644) Multiply({Tile: v17:U32}, { convertForwardSubsetTensor_7_stride_1_8: s24:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 644): v16
v_mul_lo_u32 v16, v17, s24 // (op 644) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 644): v17

 // (op 644) BEGIN: Split
 // (op 644) {Split: Add({Tile: v19:U32}, v16:U32)U32}
// Allocated DataFlowTag547: 1 VGPR (Value: UInt32) (op 644): v17
v_add_u32 v17, v19, v16 // (op 644) 
 // (op 644) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 644): v16

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 644): v19

 // (op 444) Assign VGPR {Split: Add({Tile: ShiftLAdd(LogicalShiftR(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 1:U32)U32, 2:U32, {Flatten[1]Tile: Subtract(DataFlowTag(268)U32, BitwiseAnd(DataFlowTag(268)U32, -2:I)U32)U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(433)U32, 5:U32, LogicalShiftR(DataFlowTag(268)U32, 1:I)U32)U32}, Convert(Tensor_7_stride_1_8:I64)U32)U32)U32}(644) END
 // (op 444) Assign VGPR 2:U64(645) BEGIN
 // (op 645) Assign dim(548) = 2:U64
 // (op 645) Assign stride expression
 // (op 444) Assign VGPR 2:U64(645) END
 // (op 444) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(646) BEGIN
 // (op 646) Assign dim(550) = Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB
 // (op 646) Generate Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag550: SGPR Buffer: None x 1: s[52:55]
 // (op 646) reg expression
 // (op 646) reg expression
 // (op 646) Multiply({Tensor_7_stride_1_8: s[24:25]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 646) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 646)           lhs (Tensor_7_stride_1_8: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 646)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 646): s0, s1
s_mul_i32 s1, s24, s13 // (op 646) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 646): s14
s_mul_i32 s14, s25, s12 // (op 646) most significant: low of high * low
s_mul_i32 s0, s24, s12 // (op 646) least significant
// Allocated : 1 SGPR (Value: Int32) (op 646): s15
s_mul_hi_u32 s15, s24, s12 // (op 646) most significant: high of low * low
s_add_u32 s1, s1, s14 // (op 646) most significant: sum
s_add_u32 s1, s1, s15 // (op 646) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 646): s15

 // Freeing : 1 SGPR (Value: Int32) (op 646): s14

 // (op 646) reg expression
 // (op 646) reg expression
s_mov_b32 s52, s26 // (op 646) reg expression
s_mov_b32 s53, s27 // (op 646) reg expression
s_mov_b32 s54, s0 // (op 646) reg expression
s_mov_b32 s55, 131072 // (op 646) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 646): s0, s1

 // (op 444) Assign SGPR Concatenate(Tensor_7_pointer:PG, BitFieldExtract(Multiply(Tensor_7_stride_1_8:I64, Tensor_2_size_1_4:I64)I64, width:32, offset:0)R, 131072:R)PB(646) END
 // (op 444) NOP(637) BEGIN
 // (op 637) generate({})
 // (op 637) end: generate({})
 // (op 444) NOP(637) END
 // (op 444) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(651) BEGIN
 // (op 651) Assign dim(552) = Convert(Tensor_7_stride_1_8:I64)U64
 // (op 651) Assign stride expression
 // (op 444) Assign VGPR Convert(Tensor_7_stride_1_8:I64)U64(651) END
 // (op 444) NOP(638) BEGIN
 // (op 638) generate({})
 // (op 638) end: generate({})
 // (op 444) NOP(638) END
 // (op 444) Assign VGPR 1:U64(654) BEGIN
 // (op 654) Assign dim(554) = 1:U64
 // (op 654) Assign stride expression
 // (op 444) Assign VGPR 1:U64(654) END
 // (op 444) NOP(732) BEGIN
 // (op 732) generate({})
 // (op 732) end: generate({})
 // (op 444) NOP(732) END
 // (op 444) Assign VGPR {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32}(737) BEGIN
 // (op 737) Assign dim(576) = {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32}
 // tag 576: v**UNALLOCATED**
 // (op 737) Generate {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32} into DataFlowTag576: VGPR Value: UInt32 x 1: (unallocated)
 // (op 737) reg expression
 // (op 737) BitwiseAnd({Workitem Index X: v1:U32}, -4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 737): v16
v_and_b32 v16, -4, v1 // (op 737) 
 // (op 737) BEGIN: Tile[1]
 // (op 737) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 737): v19
v_sub_u32 v19, v1, v16 // (op 737) 
 // (op 737) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 737): v16

 // (op 737) BEGIN: Flatten
 // (op 737) {Flatten: ShiftL({Tile[1]: v19:U32}, 4:U32)U32}
 // (op 737) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 737)         value (Tile[1]: VGPR Value: UInt32 x 1: v19) 
 // (op 737)         shiftAmount (Literal Value: UInt32 x 0: 4)
// Allocated : 1 VGPR (Value: UInt32) (op 737): v16
v_lshlrev_b32 v16, 4, v19 // (op 737) 
 // (op 737) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 737): v19

 // (op 737) LogicalShiftR({Workitem Index X: v1:U32}, 2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 737): v19
v_lshrrev_b32 v19, 2, v1 // (op 737) 
 // (op 737) BEGIN: Flatten
// Allocated DataFlowTag576: 1 VGPR (Value: UInt32) (op 737): v20
v_lshl_add_u32 v20, v19, 7, v16 // (op 737) 
 // (op 737) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 737): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 737): v19

 // (op 444) Assign VGPR {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(91)U32, 2:I)U32, 7:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(91)U32, BitwiseAnd(DataFlowTag(91)U32, -4:I)U32)U32}, 4:U32)U32})U32}(737) END
 // (op 444) Assign VGPR 64:U32(738) BEGIN
 // (op 738) Assign dim(577) = 64:U32
 // (op 738) Assign stride expression
 // (op 444) Assign VGPR 64:U32(738) END
 // (op 444) NOP(733) BEGIN
 // (op 733) generate({})
 // (op 733) end: generate({})
 // (op 444) NOP(733) END
 // (op 444) Assign VGPR 1:U32(742) BEGIN
 // (op 742) Assign dim(579) = 1:U32
 // (op 742) Assign stride expression
 // (op 444) Assign VGPR 1:U32(742) END
 // (op 444) NOP(785) BEGIN
 // (op 785) generate({})
 // (op 785) end: generate({})
 // (op 444) NOP(785) END
 // (op 444) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(790) BEGIN
 // (op 790) Assign dim(592) = LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32
 // tag 592: v**UNALLOCATED**
 // (op 790) Generate LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32 into DataFlowTag592: VGPR Value: UInt32 x 1: (unallocated)
 // (op 790) reg expression
 // (op 790) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 790): v16
v_and_b32 v16, -2, v1 // (op 790) 
 // (op 790) BEGIN: Tile[1]
 // (op 790) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 790): v19
v_sub_u32 v19, v1, v16 // (op 790) 
 // (op 790) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 790): v16

 // (op 790) BEGIN: Flatten
 // (op 790) {Flatten: ShiftL({Tile[1]: v19:U32}, 5:U32)U32}
 // (op 790) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 790)         value (Tile[1]: VGPR Value: UInt32 x 1: v19) 
 // (op 790)         shiftAmount (Literal Value: UInt32 x 0: 5)
// Allocated : 1 VGPR (Value: UInt32) (op 790): v16
v_lshlrev_b32 v16, 5, v19 // (op 790) 
 // (op 790) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 790): v19

 // (op 790) LogicalShiftR({Workitem Index X: v1:U32}, 1:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 790): v19
v_lshrrev_b32 v19, 1, v1 // (op 790) 
 // (op 790) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 790): v21
v_lshl_add_u32 v21, v19, 6, v16 // (op 790) 
 // (op 790) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 790): v16

 // Freeing : 1 VGPR (Value: UInt32) (op 790): v19

 // (op 790) ShiftL({Flatten: v21:U32}, 2:U32)U32
 // (op 790) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 790)         value (Flatten: VGPR Value: UInt32 x 1: v21) 
 // (op 790)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 790): v16
v_lshlrev_b32 v16, 2, v21 // (op 790) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 790): v21

 // (op 790) LogicalShiftR(v16:U32, 3:U32)U32
// Allocated DataFlowTag592: 1 VGPR (Value: UInt32) (op 790): v19
v_lshrrev_b32 v19, 3, v16 // (op 790) 
 // Freeing : 1 VGPR (Value: UInt32) (op 790): v16

 // (op 444) Assign VGPR LogicalShiftR(ShiftL({Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(157)U32, 1:I)U32, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract(DataFlowTag(157)U32, BitwiseAnd(DataFlowTag(157)U32, -2:I)U32)U32}, 5:U32)U32})U32}, 2:U32)U32, 3:U32)U32(790) END
 // (op 444) Assign VGPR 32:U32(791) BEGIN
 // (op 791) Assign dim(593) = 32:U32
 // (op 791) Assign stride expression
 // (op 444) Assign VGPR 32:U32(791) END
 // (op 444) NOP(786) BEGIN
 // (op 786) generate({})
 // (op 786) end: generate({})
 // (op 444) NOP(786) END
 // (op 444) Assign VGPR 0:U32(795) BEGIN
 // (op 795) Assign dim(595) = 0:U32
 // (op 795) Assign stride expression
 // (op 444) Assign VGPR 0:U32(795) END
 // (op 444) NOP(838) BEGIN
 // (op 838) generate({})
 // (op 838) end: generate({})
 // (op 444) NOP(838) END
 // (op 444) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(843) BEGIN
 // (op 843) Assign dim(608) = {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}
 // tag 608: v**UNALLOCATED**
 // (op 843) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32} into DataFlowTag608: VGPR Value: UInt32 x 1: (unallocated)
 // (op 843) reg expression
 // (op 843) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 843): v16
v_and_b32 v16, -2, v1 // (op 843) 
 // (op 843) BEGIN: Tile[1]Flatten
 // (op 843) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 843): v21
v_sub_u32 v21, v1, v16 // (op 843) 
 // (op 843) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 843): v16

 // (op 843) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 843): v16
v_and_b32 v16, 4294967294, v1 // (op 843) 
 // (op 843) BEGIN: Flatten
 // (op 843) {Flatten: Add(v16:U32, {Tile[1]Flatten: v21:U32})U32}
// Allocated DataFlowTag608: 1 VGPR (Value: UInt32) (op 843): v22
v_add_u32 v22, v16, v21 // (op 843) 
 // (op 843) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 843): v21

 // Freeing : 1 VGPR (Value: UInt32) (op 843): v16

 // (op 444) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(223)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(223)U32, BitwiseAnd(DataFlowTag(223)U32, -2:I)U32)U32})U32}(843) END
 // (op 444) Assign VGPR 2:U32(844) BEGIN
 // (op 844) Assign dim(609) = 2:U32
 // (op 844) Assign stride expression
 // (op 444) Assign VGPR 2:U32(844) END
 // (op 444) NOP(839) BEGIN
 // (op 839) generate({})
 // (op 839) end: generate({})
 // (op 444) NOP(839) END
 // (op 444) Assign VGPR 1:U32(848) BEGIN
 // (op 848) Assign dim(611) = 1:U32
 // (op 848) Assign stride expression
 // (op 444) Assign VGPR 1:U32(848) END
 // (op 444) NOP(891) BEGIN
 // (op 891) generate({})
 // (op 891) end: generate({})
 // (op 444) NOP(891) END
 // (op 444) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(896) BEGIN
 // (op 896) Assign dim(624) = {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}
 // tag 624: v**UNALLOCATED**
 // (op 896) Generate {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32} into DataFlowTag624: VGPR Value: UInt32 x 1: (unallocated)
 // (op 896) reg expression
 // (op 896) BitwiseAnd({Workitem Index X: v1:U32}, -2:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 896): v16
v_and_b32 v16, -2, v1 // (op 896) 
 // (op 896) BEGIN: Tile[1]Flatten
 // (op 896) {Tile[1]Flatten: Subtract({Workitem Index X: v1:U32}, v16:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 896): v21
v_sub_u32 v21, v1, v16 // (op 896) 
 // (op 896) END: Tile[1]Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 896): v16

 // (op 896) BitwiseAnd({Workitem Index X: v1:U32}, 4294967294:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 896): v16
v_and_b32 v16, 4294967294, v1 // (op 896) 
 // (op 896) BEGIN: Flatten
 // (op 896) {Flatten: Add(v16:U32, {Tile[1]Flatten: v21:U32})U32}
// Allocated DataFlowTag624: 1 VGPR (Value: UInt32) (op 896): v23
v_add_u32 v23, v16, v21 // (op 896) 
 // (op 896) END: Flatten
 // Freeing Tile[1]Flatten: 1 VGPR (Value: UInt32) (op 896): v21

 // Freeing : 1 VGPR (Value: UInt32) (op 896): v16

 // (op 444) Assign VGPR {Flatten: Add(BitwiseAnd(DataFlowTag(283)U32, 4294967294:U32)U32, {Tile[1]Flatten: Subtract(DataFlowTag(283)U32, BitwiseAnd(DataFlowTag(283)U32, -2:I)U32)U32})U32}(896) END
 // (op 444) Assign VGPR 2:U32(897) BEGIN
 // (op 897) Assign dim(625) = 2:U32
 // (op 897) Assign stride expression
 // (op 444) Assign VGPR 2:U32(897) END
 // (op 444) NOP(892) BEGIN
 // (op 892) generate({})
 // (op 892) end: generate({})
 // (op 444) NOP(892) END
 // (op 444) Assign VGPR 1:U32(901) BEGIN
 // (op 901) Assign dim(627) = 1:U32
 // (op 901) Assign stride expression
 // (op 444) Assign VGPR 1:U32(901) END
 // (op 444) ForLoopOp KLoopTail: LessThan(DataFlowTag(448)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL(171) BEGIN
 // (op 171) Initialize For Loop
 // (op 171) generate({172})
 // (op 171) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32(172) BEGIN
 // (op 172) Assign dim(448) = BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32
 // tag 448: s**UNALLOCATED**
 // (op 172) Generate BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32 into DataFlowTag448: SGPR Value: UInt32 x 1: (unallocated)
 // (op 172) reg expression
 // (op 172) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated  convertForwardSubset: 2 SGPRs (Value: Int64) (op 172): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 172) 
 // (op 172) convert
 // (op 172) BitwiseAnd({ convertForwardSubset: s0:U32}, 4294967294:U32)U32
// Allocated DataFlowTag448: 1 SGPR (Value: UInt32) (op 172): s14
s_and_b32 s14, s0, 4294967294 // (op 172) 
 // Freeing  convertForwardSubset: 2 SGPRs (Value: Int64) (op 172): s0, s1

 // (op 171) Assign SGPR BitwiseAnd(Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32, 4294967294:U32)U32(172) END
 // (op 171) end: generate({172})
 // (op 171) Lock For Loop
 // Lock Branch
// DEBUG: Wait for scalar queue
 // (op 171) 
 // (op 171) Generate LessThan(DataFlowTag(448)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 171) LessThan({DataFlowTag448: s14:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32)BL
 // (op 171) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 171): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 171) 
// Allocated : 1 SGPR (Value: UInt32) (op 171): s15
s_mov_b32 s15, s0 // (op 171) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 171): s0, s1

s_cmp_lt_u32 s14, s15 // (op 171) 
 // Freeing : 1 SGPR (Value: UInt32) (op 171): s15

 // Freeing : 4 VGPRs (Value: FP4x8) (op 378): v36, v37, v38, v39

 // Freeing : 1 VGPR (Value: E8M0) (op 381): v12

 // Freeing : 8 VGPRs (Value: BF8x4) (op 384): v40, v41, v42, v43, v44, v45, v46, v47

 // Freeing : 1 VGPR (Value: E8M0) (op 387): v14

s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before branching into Bottom of TailLoop!
 // (op 171) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 171) 
s_cbranch_scc0 RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 // (op 171) Condition: Top (jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171 if false)
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before falling through to TailLoop!
 // (op 171) 
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171:
 // (op 171) 
 // (op 171) generate({173, 222})
 // (op 171) Assign SGPR Add(DataFlowTag(448)U32, 1:U32)U32(173) BEGIN
 // (op 173) Assign dim(448) = Add(DataFlowTag(448)U32, 1:U32)U32
 // (op 173) Generate Add(DataFlowTag(448)U32, 1:U32)U32 into DataFlowTag448: SGPR Value: UInt32 x 1: s14
 // (op 173) reg expression
 // (op 173) Add({DataFlowTag448: s14:U32}, 1:U32)U32
 // (op 171) SetCoordinate(222) BEGIN
 // (op 222) generate({})
 // (op 222) end: generate({})
 // (op 222) generate({176})
 // (op 222) LoadTiled Value: BF8x4(176) BEGIN
 // (op 176) GEN: loadMacroTileVGPRCI Value: BF8x4
 // (op 176) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 176) Generate Convert(Tensor_0_stride_0_0:I64)U64 into Stride495: VGPR Value: UInt64 x 1: (unallocated)
 // (op 176) reg expression
s_add_u32 s14, s14, 1 // (op 173) 
 // (op 171) Assign SGPR Add(DataFlowTag(448)U32, 1:U32)U32(173) END
// Allocated Stride495: 2 VGPRs (Value: UInt64) (op 176): v24, v25
v_mov_b64 v[24:25], s[2:3] // (op 176) convert
 // (op 176) Generate 1:U64 into nullptr
 // tag 450: v**UNALLOCATED**
 // (op 176) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Stride495Tensor_0_stride_0_0: VGPR Value: UInt64 x 1: v[24:25]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(Convert(Multiply(16:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
 // 	t.trLoadPairStride = Add(Multiply(Convert(Multiply(16:U32, CommandArgument(Tensor_0_stride_0)I64)I64)U64, 1:U32)U64, 0:U32)U64
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
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 494: v**UNALLOCATED**
// Allocated Offset494: 1 VGPR (Value: UInt32) (op 176): v12
v_mov_b32 v12, v15 // (op 176) 
// Allocated : 1 VGPR (Value: UInt32) (op 176): v14
v_mov_b32 v14, v12 // (op 176) 
 // (op 176)   M 2 N 4 elementsPerMove 4 bytesPerMove 16 rowStride {Stride495Tensor_0_stride_0_0: v[24:25]:U64} colStride 1:U64 vgprBlockSize 16 numVGPRBlocks 1
// Allocated : 8 VGPRs (Value: BF8x4) (op 176): v26, v27, v28, v29, v30, v31, v32, v33
buffer_load_dwordx4 v[26:29], v14, s[40:43], 0 offen // (op 176) Load value
// VMEM: Expected complete at 4326 (current 3846)
 // (op 176) Generate Add(v14:U32, v24:R)U32 into VGPR Value: UInt32 x 1: v14
 // (op 176) reg expression
 // (op 176) reg expression
 // (op 176) Add(v14:U32, v24:R)U32
v_add_u32 v14, v14, v24 // (op 176) 
buffer_load_dwordx4 v[30:33], v14, s[40:43], 0 offen // (op 176) Load value
// VMEM: Expected complete at 4328 (current 3848)
 // Freeing Stride495: 2 VGPRs (Value: UInt64) (op 176): v24, v25

 // Freeing : 1 VGPR (Value: UInt32) (op 176): v14

 // (op 222) LoadTiled Value: BF8x4(176) END
 // (op 222) Deallocate{}(1196) BEGIN
 // (op 1196) Deallocate 494
 // Freeing Offset494: 1 VGPR (Value: UInt32) (op 176): v12

 // Deleting tag 494
 // (op 1196) Deallocate 496
 // Deleting tag 496
 // (op 222) Deallocate{}(1196) END
 // (op 222) Assign VGPR Add(DataFlowTag(490)U32, Convert(DataFlowTag(491)U64)U32)U32(439) BEGIN
 // (op 439) Assign dim(490) = Add(DataFlowTag(490)U32, Convert(DataFlowTag(491)U64)U32)U32
 // (op 439) Generate Add(DataFlowTag(490)U32, Convert(DataFlowTag(491)U64)U32)U32 into DataFlowTag490: VGPR Value: UInt32 x 1: v15
 // (op 439) reg expression
 // (op 439) Add({DataFlowTag490: v15:U32}, 64:U32)U32
v_add_u32 v15, 64, v15 // (op 439) 
 // (op 222) Assign VGPR Add(DataFlowTag(490)U32, Convert(DataFlowTag(491)U64)U32)U32(439) END
 // (op 222) end: generate({176})
 // (op 171) SetCoordinate(222) END
 // (op 171) SetCoordinate(227) BEGIN
 // (op 227) generate({})
 // (op 227) end: generate({})
 // (op 227) generate({190})
 // (op 227) LoadTiled Value: FP4x8(190) BEGIN
 // (op 190) GEN: loadMacroTileVGPRCI Value: FP4x8
 // (op 190) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 190) Generate 0:U64 into nullptr
 // tag 452: v**UNALLOCATED**
 // (op 190) 	Dir = Load
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
 // (op 171) Barrier(1097) BEGIN
s_barrier  // (op 1097) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
 // (op 171) Barrier(1097) END
// Allocated Offset513: 1 VGPR (Value: UInt32) (op 190): v12
v_mov_b32 v12, v13 // (op 190) 
 // (op 190) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
 // 	info.rowOffsetReg = Offset513: VGPR Value: UInt32 x 1: v12
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
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

 // (op 171) StoreLDSTile Value: BF8x4(177) BEGIN
 // (op 177) GEN: StoreLDSTile
 // (op 177) GEN: storeMacroTileLDS OP 177 LDS 17 MacroTile 450
 // (op 177) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 4

 // (op 177) Generate 64:U32 into nullptr
 // (op 177) Generate 1:U32 into nullptr
 // (op 177) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[26:33]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 64
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1088
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

 // (op 177) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 2
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 4
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: BF8x4 x 8: v[26:33]
 // 	info.rowOffsetReg = DataFlowTag576: VGPR Value: UInt32 x 1: v20
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 64
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(1024:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 1
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 16
 // 	t.elementBlockStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Multiply(16:U32, 1:U32)U32, 0:U32)U32
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 1088
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 177) 	info.m = 2
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 64
 // 	colStride = 1
 // 	info.colStrideAttributes.elementBlockSize = 16
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 4 VGPRs (Value: FP4x8) (op 190): v34, v35, v36, v37
buffer_load_dwordx4 v[34:37], v12, s[44:47], 0 offen // (op 190) Load value
// VMEM: Expected complete at 4332 (current 3852)
 // (op 227) LoadTiled Value: FP4x8(190) END
 // (op 227) Deallocate{}(1208) BEGIN
 // (op 1208) Deallocate 513
 // Freeing Offset513: 1 VGPR (Value: UInt32) (op 190): v12

 // Deleting tag 513
 // (op 1208) Deallocate 515
 // Deleting tag 515
 // (op 227) Deallocate{}(1208) END
 // (op 227) Assign VGPR Add(DataFlowTag(509)U32, Convert(DataFlowTag(510)U64)U32)U32(508) BEGIN
 // (op 508) Assign dim(509) = Add(DataFlowTag(509)U32, Convert(DataFlowTag(510)U64)U32)U32
 // (op 508) Generate Add(DataFlowTag(509)U32, Convert(DataFlowTag(510)U64)U32)U32 into DataFlowTag509: VGPR Value: UInt32 x 1: v13
 // (op 508) reg expression
 // (op 508) Add({DataFlowTag509: v13:U32}, 32:U32)U32
v_add_u32 v13, 32, v13 // (op 508) 
 // (op 227) Assign VGPR Add(DataFlowTag(509)U32, Convert(DataFlowTag(510)U64)U32)U32(508) END
 // (op 227) end: generate({190})
 // (op 171) SetCoordinate(227) END
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[26:29] offset:1088 // (op 177) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 128, 144, 160, 176, 256, 272, 288, 304, 384, 400, 416, 432, 512, 528, 544, 560, 640, 656, 672, 688, 768, 784, 800, 816, 896, 912, 928, 944, 1024, 1040, 1056, 1072, 1152, 1168, 1184, 1200, 1280, 1296, 1312, 1328, 1408, 1424, 1440, 1456, 1536, 1552, 1568, 1584, 1664, 1680, 1696, 1712, 1792, 1808, 1824, 1840, 1920, 1936, 1952, 1968]
// DSMEM: Expected complete at 4019 (current 3924)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
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
// Allocated Offset532: 1 VGPR (Value: UInt32) (op 191): v12
v_mov_b32 v12, v18 // (op 191) 
 // (op 191) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset532: VGPR Value: UInt32 x 1: v12
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

// Allocated : 1 VGPR (Value: E8M0) (op 191): v14
buffer_load_ubyte v14, v12, s[48:51], 0 offen offset: 0 // (op 191) Load value
// VMEM: Expected complete at 4810 (current 4330)
 // (op 232) LoadTiled Value: E8M0(191) END
 // (op 232) Deallocate{}(1210) BEGIN
 // (op 1210) Deallocate 532
 // Freeing Offset532: 1 VGPR (Value: UInt32) (op 191): v12

 // Deleting tag 532
 // (op 1210) Deallocate 534
 // Deleting tag 534
 // (op 232) Deallocate{}(1210) END
 // (op 232) Assign VGPR Add(DataFlowTag(528)U32, Convert(DataFlowTag(529)U64)U32)U32(572) BEGIN
 // (op 572) Assign dim(528) = Add(DataFlowTag(528)U32, Convert(DataFlowTag(529)U64)U32)U32
 // (op 572) Generate Add(DataFlowTag(528)U32, Convert(DataFlowTag(529)U64)U32)U32 into DataFlowTag528: VGPR Value: UInt32 x 1: v18
 // (op 572) reg expression
 // (op 572) Add({DataFlowTag528: v18:U32}, 2:U32)U32
v_add_u32 v18, 2, v18 // (op 572) 
 // (op 232) Assign VGPR Add(DataFlowTag(528)U32, Convert(DataFlowTag(529)U64)U32)U32(572) END
 // (op 232) end: generate({191})
 // (op 171) SetCoordinate(232) END
s_waitcnt vmcnt(2)
ds_write_b128 v20, v[30:33] offset:1152 // (op 177) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 128, 144, 160, 176, 256, 272, 288, 304, 384, 400, 416, 432, 512, 528, 544, 560, 640, 656, 672, 688, 768, 784, 800, 816, 896, 912, 928, 944, 1024, 1040, 1056, 1072, 1152, 1168, 1184, 1200, 1280, 1296, 1312, 1328, 1408, 1424, 1440, 1456, 1536, 1552, 1568, 1584, 1664, 1680, 1696, 1712, 1792, 1808, 1824, 1840, 1920, 1936, 1952, 1968]
// DSMEM: Expected complete at 4024 (current 3929)
// Extra dsts: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
 // (op 171) StoreLDSTile Value: BF8x4(177) END
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
 // (op 171) Deallocate{}(1198) BEGIN
 // (op 1198) Deallocate 578
 // Deleting tag 578
 // (op 1198) Deallocate 450
 // Freeing : 8 VGPRs (Value: BF8x4) (op 176): v26, v27, v28, v29, v30, v31, v32, v33

 // Deleting tag 450
 // (op 171) Deallocate{}(1198) END
// Allocated Offset551: 1 VGPR (Value: UInt32) (op 192): v12
v_mov_b32 v12, v17 // (op 192) 
 // (op 192) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 8
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: E8M0 x 1: (unallocated)
 // 	info.rowOffsetReg = Offset551: VGPR Value: UInt32 x 1: v12
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

 // (op 171) Barrier(1110) BEGIN
s_barrier  // (op 1110) 
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 0, Size: 1024)
 // (op 171) Barrier(1110) END
// Allocated : 1 VGPR (Value: E8M0) (op 192): v16
buffer_load_ubyte v16, v12, s[52:55], 0 offen offset: 0 // (op 192) Load value
// VMEM: Expected complete at 4816 (current 4336)
 // (op 237) LoadTiled Value: E8M0(192) END
 // (op 237) Deallocate{}(1212) BEGIN
 // (op 1212) Deallocate 553
 // Deleting tag 553
 // (op 1212) Deallocate 551
 // Freeing Offset551: 1 VGPR (Value: UInt32) (op 192): v12

 // Deleting tag 551
 // (op 237) Deallocate{}(1212) END
 // (op 237) Assign VGPR Add(DataFlowTag(547)U32, Convert(DataFlowTag(548)U64)U32)U32(636) BEGIN
 // (op 636) Assign dim(547) = Add(DataFlowTag(547)U32, Convert(DataFlowTag(548)U64)U32)U32
 // (op 636) Generate Add(DataFlowTag(547)U32, Convert(DataFlowTag(548)U64)U32)U32 into DataFlowTag547: VGPR Value: UInt32 x 1: v17
 // (op 636) reg expression
 // (op 636) Add({DataFlowTag547: v17:U32}, 2:U32)U32
 // (op 171) StoreLDSTile Value: FP4x8(187) BEGIN
 // (op 187) GEN: StoreLDSTile
 // (op 187) GEN: storeMacroTileLDS OP 187 LDS 25 MacroTile 452
 // (op 187) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 187) Generate 0:U32 into nullptr
 // (op 187) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[34:37]
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

 // (op 187) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: v[34:37]
 // 	info.rowOffsetReg = DataFlowTag592: VGPR Value: UInt32 x 1: v19
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
 // (op 187) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

v_add_u32 v17, 2, v17 // (op 636) 
 // (op 237) Assign VGPR Add(DataFlowTag(547)U32, Convert(DataFlowTag(548)U64)U32)U32(636) END
 // (op 237) end: generate({192})
 // (op 171) SetCoordinate(237) END
s_waitcnt vmcnt(2)
ds_write_b128 v19, v[34:37] // (op 187) Store local data 
// Modelled addresses (normalized): [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688, 704, 720, 736, 752, 768, 784, 800, 816, 832, 848, 864, 880, 896, 912, 928, 944, 960, 976, 992, 1008]
// DSMEM: Expected complete at 4030 (current 3935)
// Extra dsts: LDS Value: FP4x8 x 0: LDS:(Offset: 0, Size: 1024)
 // (op 171) StoreLDSTile Value: FP4x8(187) END
 // (op 171) Barrier(1100) BEGIN
 // (op 171) Deallocate{}(1202) BEGIN
 // (op 1202) Deallocate 594
 // Deleting tag 594
 // (op 1202) Deallocate 452
 // Freeing : 4 VGPRs (Value: FP4x8) (op 190): v34, v35, v36, v37

 // Deleting tag 452
 // (op 171) Deallocate{}(1202) END
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1100) 
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
 // (op 171) Barrier(1100) END
 // (op 171) Barrier(1124) BEGIN
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
 // 	info.offset = Literal Value: UInt32 x 0: 1088
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
 // 	info.offset = Literal Value: UInt32 x 0: 1088
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

s_barrier  // (op 1124) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 3136, Size: 64)
 // (op 171) Barrier(1124) END
// Allocated : 8 VGPRs (Value: BF8x4) (op 179): v24, v25, v26, v27, v28, v29, v30, v31
ds_read_b128 v[24:27], v3 offset:1088 // (op 179) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4123 (current 4028)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
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
 // 	info.offset = Literal Value: UInt32 x 0: 3136
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
 // 	info.data = VGPR Value: E8M0 x 1: v14
 // 	info.rowOffsetReg = DataFlowTag608: VGPR Value: UInt32 x 1: v22
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
 // 	info.offset = Literal Value: UInt32 x 0: 3136
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

ds_read_b128 v[28:31], v3 offset:1120 // (op 179) Load local data 
// Modelled addresses (normalized): [0, 64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 16, 80, 144, 208, 272, 336, 400, 464, 528, 592, 656, 720, 784, 848, 912, 976, 1040, 1104, 1168, 1232, 1296, 1360, 1424, 1488, 1552, 1616, 1680, 1744, 1808, 1872, 1936, 2000]
// DSMEM: Expected complete at 4124 (current 4029)
// Extra srcs: LDS Value: BF8x4 x 0: LDS:(Offset: 1088, Size: 2048)
 // (op 178) LoadLDSTile{Value: BF8x4}(179) END
 // (op 178) end: generate({179})
 // (op 171) SetCoordinate(178) END
s_waitcnt vmcnt(1)
ds_write_b8 v22, v14 offset:3136 // (op 188) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected stall of 1. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 4127 (current 4032)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 3136, Size: 64)
 // (op 171) StoreLDSTile Value: E8M0(188) END
 // (op 171) Barrier(1114) BEGIN
 // (op 171) Deallocate{}(1204) BEGIN
 // (op 1204) Deallocate 610
 // Deleting tag 610
 // (op 1204) Deallocate 454
 // Freeing : 1 VGPR (Value: E8M0) (op 191): v14

 // Deleting tag 454
 // (op 171) Deallocate{}(1204) END
s_waitcnt lgkmcnt(3)
s_barrier  // (op 1114) 
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 0, Size: 1024)
 // (op 171) Barrier(1114) END
 // (op 171) Barrier(1138) BEGIN
 // (op 171) SetCoordinate(180) BEGIN
 // (op 180) generate({})
 // (op 180) end: generate({})
 // (op 180) generate({181})
 // (op 180) LoadLDSTile{Value: FP4x8}(181) BEGIN
 // (op 181) GEN: loadMacroTileWAVELDS OP 181 LDS 25 WaveTile 169
 // (op 181) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 8

 // (op 181) Generate 0:U32 into nullptr
 // tag 460: v**UNALLOCATED**
 // (op 181) 	Dir = Load
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

 // (op 181) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 8
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: FP4x8 x 4: (unallocated)
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
 // 	t.elementBlockStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
 // 	t.trLoadPairStride = Add(Divide(Multiply(64:U32, 4:U32)U32, 8:U32)U32, 0:U32)U32
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
 // (op 181) 	info.m = 1
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 0
 // 	colStride = 0
 // 	info.colStrideAttributes.elementBlockSize = 32
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_barrier  // (op 1138) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 1024, Size: 64)
 // (op 171) Barrier(1138) END
// Allocated : 4 VGPRs (Value: FP4x8) (op 181): v32, v33, v34, v35
ds_read_b128 v[32:35], v4 // (op 181) Load local data 
// Modelled addresses (normalized): [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992, 16, 48, 80, 112, 144, 176, 208, 240, 272, 304, 336, 368, 400, 432, 464, 496, 528, 560, 592, 624, 656, 688, 720, 752, 784, 816, 848, 880, 912, 944, 976, 1008]
// DSMEM: Expected stall of 88. CBNW: 0, Inc: 3
// DSMEM: Expected complete at 4219 (current 4124)
// Extra srcs: LDS Value: FP4x8 x 0: LDS:(Offset: 0, Size: 1024)
 // (op 180) LoadLDSTile{Value: FP4x8}(181) END
 // (op 180) end: generate({181})
 // (op 171) SetCoordinate(180) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 1024
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
 // 	info.data = VGPR Value: E8M0 x 1: v16
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
 // 	info.offset = Literal Value: UInt32 x 0: 1024
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

 // (op 171) Barrier(1128) BEGIN
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1128) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 3136, Size: 64)
 // (op 171) Barrier(1128) END
s_waitcnt vmcnt(0)
ds_write_b8 v23, v16 offset:1024 // (op 189) Store local data 
// Modelled addresses (normalized): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63]
// DSMEM: Expected complete at 4226 (current 4131)
// Extra dsts: LDS Value: E8M0 x 0: LDS:(Offset: 1024, Size: 64)
 // (op 171) StoreLDSTile Value: E8M0(189) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 3136
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
 // 	info.offset = Literal Value: UInt32 x 0: 3136
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

 // (op 171) Deallocate{}(1206) BEGIN
 // (op 1206) Deallocate 456
 // Freeing : 1 VGPR (Value: E8M0) (op 192): v16

 // Deleting tag 456
 // (op 1206) Deallocate 626
 // Deleting tag 626
 // (op 171) Deallocate{}(1206) END
// Allocated : 1 VGPR (Value: E8M0) (op 183): v12
ds_read_u8 v12, v0 offset:3136 // (op 183) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4227 (current 4132)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 3136, Size: 64)
 // (op 182) LoadLDSTile{Value: E8M0}(183) END
 // (op 182) end: generate({183})
 // (op 171) SetCoordinate(182) END
 // (op 171) Barrier(1142) BEGIN
s_waitcnt lgkmcnt(1)
s_barrier  // (op 1142) 
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 1024, Size: 64)
 // (op 171) Barrier(1142) END
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
 // 	info.offset = Literal Value: UInt32 x 0: 1024
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
 // 	info.offset = Literal Value: UInt32 x 0: 1024
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

// Allocated : 1 VGPR (Value: E8M0) (op 185): v14
ds_read_u8 v14, v6 offset:1024 // (op 185) Load local data 
// Modelled addresses (normalized): [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 61, 63]
// DSMEM: Expected complete at 4324 (current 4229)
// Extra srcs: LDS Value: E8M0 x 0: LDS:(Offset: 1024, Size: 64)
 // (op 184) LoadLDSTile{Value: E8M0}(185) END
 // (op 184) end: generate({185})
 // (op 171) SetCoordinate(184) END
 // (op 171) Multiply(186) BEGIN
 // (op 186) Generate ScaledMatrixMultiply(WaveTile4xBF8, WaveTile8xFP4, {DataFlowTag55: a[0:15]:S}, WaveTileE8M0, WaveTileE8M0)S into DataFlowTag55: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[24:31], v[32:35], a[0:15], v12, v14 op_sel:[0,0] op_sel_hi:[0,0] cbsz:0b001 blgp:0b100 // (op 186) 
 // (op 171) Multiply(186) END
 // (op 171) Deallocate{}(1200) BEGIN
 // (op 1200) Deallocate 458
 // Freeing : 8 VGPRs (Value: BF8x4) (op 179): v24, v25, v26, v27, v28, v29, v30, v31

 // Deleting tag 458
 // (op 1200) Deallocate 460
 // Freeing : 4 VGPRs (Value: FP4x8) (op 181): v32, v33, v34, v35

 // Deleting tag 460
 // (op 1200) Deallocate 462
 // Freeing : 1 VGPR (Value: E8M0) (op 183): v12

 // Deleting tag 462
 // (op 1200) Deallocate 464
 // Freeing : 1 VGPR (Value: E8M0) (op 185): v14

 // Deleting tag 464
 // (op 171) Deallocate{}(1200) END
 // (op 171) end: generate({173, 222})
 // (op 171) For Loop Increment
 // (op 171) generate({})
 // (op 171) end: generate({})
 // (op 171) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 if true)
 // (op 171) Generate LessThan(DataFlowTag(448)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 171) LessThan({DataFlowTag448: s14:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64)U32)BL
 // (op 171) ArithmeticShiftR({Tensor_0_size_1_14: s[28:29]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 171): s0, s1
s_ashr_i64 s[0:1], s[28:29], 6 // (op 171) 
// Allocated : 1 SGPR (Value: UInt32) (op 171): s15
s_mov_b32 s15, s0 // (op 171) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 171): s0, s1

s_cmp_lt_u32 s14, s15 // (op 171) 
 // Freeing : 1 SGPR (Value: UInt32) (op 171): s15

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 171) 
s_cbranch_scc1 RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 // (op 171) Condition: Bottom (jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopTop_KLoopTail_171 if true)
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ForLoopBottom_KLoopTail_171:
 // (op 171) 
 // (op 171) Unlock For Loop
 // Unlock None
 // (op 444) ForLoopOp KLoopTail: LessThan(DataFlowTag(448)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_14:I64, 6:U32)I64)U32)BL(171) END
 // (op 444) Deallocate{}(1194) BEGIN
 // (op 1194) Deallocate 529
 // Deleting tag 529
 // (op 1194) Deallocate 490
 // Freeing DataFlowTag490: 1 VGPR (Value: UInt32) (op 449): v15

 // Deleting tag 490
 // (op 1194) Deallocate 593
 // Deleting tag 593
 // (op 1194) Deallocate 535
 // Deleting tag 535
 // (op 1194) Deallocate 576
 // Freeing DataFlowTag576: 1 VGPR (Value: UInt32) (op 737): v20

 // Deleting tag 576
 // (op 1194) Deallocate 448
 // Freeing DataFlowTag448: 1 SGPR (Value: UInt32) (op 172): s14

 // Deleting tag 448
 // (op 1194) Deallocate 625
 // Deleting tag 625
 // (op 1194) Deallocate 577
 // Deleting tag 577
 // (op 1194) Deallocate 627
 // Deleting tag 627
 // (op 1194) Deallocate 509
 // Freeing DataFlowTag509: 1 VGPR (Value: UInt32) (op 516): v13

 // Deleting tag 509
 // (op 1194) Deallocate 611
 // Deleting tag 611
 // (op 1194) Deallocate 552
 // Deleting tag 552
 // (op 1194) Deallocate 609
 // Deleting tag 609
 // (op 1194) Deallocate 608
 // Freeing DataFlowTag608: 1 VGPR (Value: UInt32) (op 843): v22

 // Deleting tag 608
 // (op 1194) Deallocate 595
 // Deleting tag 595
 // (op 1194) Deallocate 592
 // Freeing DataFlowTag592: 1 VGPR (Value: UInt32) (op 790): v19

 // Deleting tag 592
 // (op 1194) Deallocate 516
 // Deleting tag 516
 // (op 1194) Deallocate 547
 // Freeing DataFlowTag547: 1 VGPR (Value: UInt32) (op 644): v17

 // Deleting tag 547
 // (op 1194) Deallocate 491
 // Deleting tag 491
 // (op 1194) Deallocate 497
 // Deleting tag 497
 // (op 1194) Deallocate 624
 // Freeing DataFlowTag624: 1 VGPR (Value: UInt32) (op 896): v23

 // Deleting tag 624
 // (op 1194) Deallocate 510
 // Deleting tag 510
 // (op 1194) Deallocate 514
 // Deleting tag 514
 // (op 1194) Deallocate 528
 // Freeing DataFlowTag528: 1 VGPR (Value: UInt32) (op 580): v18

 // Deleting tag 528
 // (op 1194) Deallocate 548
 // Deleting tag 548
 // (op 1194) Deallocate 495
 // Deleting tag 495
 // (op 1194) Deallocate 554
 // Deleting tag 554
 // (op 1194) Deallocate 579
 // Deleting tag 579
 // (op 1194) Deallocate 533
 // Deleting tag 533
 // (op 444) Deallocate{}(1194) END
 // (op 444) end: generate({438})
 // (op 444) Unlock Scope 444
 // Unlock None
 // (op 218) Scope(444) END
 // (op 218) end: generate({444})
 // (op 217) SetCoordinate(218) END
 // (op 217) end: generate({218})
 // (op 340) SetCoordinate(217) END
 // (op 340) Deallocate{}(1214) BEGIN
 // (op 1214) Deallocate 429
 // Deleting tag 429
 // (op 1214) Deallocate 157
 // Deleting tag 157
 // (op 1214) Deallocate 142
 // Deleting tag 142
 // (op 1214) Deallocate 586
 // Deleting tag 586
 // (op 1214) Deallocate 602
 // Deleting tag 602
 // (op 1214) Deallocate 618
 // Deleting tag 618
 // (op 1214) Deallocate 634
 // Deleting tag 634
 // (op 1214) Deallocate 4
 // Deleting tag 4
 // (op 1214) Deallocate 512
 // Freeing DataFlowTag512: 4 SGPRs (Buffer: None) (op 541): s44, s45, s46, s47

 // Deleting tag 512
 // (op 1214) Deallocate 208
 // Deleting tag 208
 // (op 1214) Deallocate 493
 // Freeing DataFlowTag493: 4 SGPRs (Buffer: None) (op 477): s40, s41, s42, s43

 // Deleting tag 493
 // (op 1214) Deallocate 427
 // Deleting tag 427
 // (op 1214) Deallocate 76
 // Deleting tag 76
 // (op 1214) Deallocate 268
 // Deleting tag 268
 // (op 1214) Deallocate 531
 // Freeing DataFlowTag531: 4 SGPRs (Buffer: None) (op 605): s48, s49, s50, s51

 // Deleting tag 531
 // (op 1214) Deallocate 223
 // Deleting tag 223
 // (op 1214) Deallocate 25
 // Deleting tag 25
 // (op 1214) Deallocate 91
 // Deleting tag 91
 // (op 1214) Deallocate 3
 // Deleting tag 3
 // (op 1214) Deallocate 1
 // Deleting tag 1
 // (op 1214) Deallocate 2
 // Deleting tag 2
 // (op 1214) Deallocate 433
 // Deleting tag 433
 // (op 1214) Deallocate 41
 // Deleting tag 41
 // (op 1214) Deallocate 33
 // Deleting tag 33
 // (op 1214) Deallocate 550
 // Freeing DataFlowTag550: 4 SGPRs (Buffer: None) (op 669): s52, s53, s54, s55

 // Deleting tag 550
 // (op 1214) Deallocate 283
 // Deleting tag 283
 // (op 1214) Deallocate 431
 // Deleting tag 431
 // (op 1214) Deallocate 17
 // Deleting tag 17
 // (op 340) Deallocate{}(1214) END
 // (op 340) Deallocate{Tensor_0_stride_0_0, Tensor_4_stride_0_6, Tensor_0_pointer, Tensor_2_pointer, Tensor_0_size_1_14, Tensor_7_stride_1_8, Tensor_0_size_0_1, Tensor_7_pointer, Tensor_4_pointer, Tensor_2_stride_1_3}(1351) BEGIN
 // (op 1351) Deallocate Tensor_0_stride_0_0
 // Freeing Tensor_0_stride_0_0: 2 SGPRs (Value: Raw32): s2, s3

 // (op 1351) Deallocate Tensor_4_stride_0_6
 // Freeing Tensor_4_stride_0_6: 2 SGPRs (Value: Raw32): s20, s21

 // (op 1351) Deallocate Tensor_0_pointer
 // Freeing Tensor_0_pointer: 2 SGPRs (Value: Raw32): s4, s5

 // (op 1351) Deallocate Tensor_2_pointer
 // Freeing Tensor_2_pointer: 2 SGPRs (Value: Raw32): s10, s11

 // (op 1351) Deallocate Tensor_0_size_1_14
 // Freeing Tensor_0_size_1_14: 2 SGPRs (Value: Raw32): s28, s29

 // (op 1351) Deallocate Tensor_7_stride_1_8
 // Freeing Tensor_7_stride_1_8: 2 SGPRs (Value: Raw32): s24, s25

 // (op 1351) Deallocate Tensor_0_size_0_1
 // Freeing Tensor_0_size_0_1: 2 SGPRs (Value: Raw32): s6, s7

 // (op 1351) Deallocate Tensor_7_pointer
 // Freeing Tensor_7_pointer: 2 SGPRs (Value: Raw32): s26, s27

 // (op 1351) Deallocate Tensor_4_pointer
 // Freeing Tensor_4_pointer: 2 SGPRs (Value: Raw32): s22, s23

 // (op 1351) Deallocate Tensor_2_stride_1_3
 // Freeing Tensor_2_stride_1_3: 2 SGPRs (Value: Raw32): s8, s9

 // (op 340) Deallocate{Tensor_0_stride_0_0, Tensor_4_stride_0_6, Tensor_0_pointer, Tensor_2_pointer, Tensor_0_size_1_14, Tensor_7_stride_1_8, Tensor_0_size_0_1, Tensor_7_pointer, Tensor_4_pointer, Tensor_2_stride_1_3}(1351) END
 // (op 340) Barrier(1216) BEGIN
s_barrier  // (op 1216) 
 // (op 340) Barrier(1216) END
 // (op 340) end: generate({345})
 // (op 340) Unlock Scope 340
 // Unlock None
 // (op 1) Scope(340) END
 // (op 1) Deallocate{}(1260) BEGIN
 // (op 1260) Deallocate 658
 // Freeing DataFlowTag658: 1 VGPR (Value: UInt32) (op 1006): v8

 // Deleting tag 658
 // (op 1260) Deallocate 650
 // Freeing DataFlowTag650: 1 VGPR (Value: UInt32) (op 982): v7

 // Deleting tag 650
 // (op 1260) Deallocate 635
 // Deleting tag 635
 // (op 1260) Deallocate 633
 // Deleting tag 633
 // (op 1260) Deallocate 584
 // Freeing DataFlowTag584: 1 VGPR (Value: UInt32) (op 765): v3

 // Deleting tag 584
 // (op 1260) Deallocate 666
 // Freeing DataFlowTag666: 1 VGPR (Value: UInt32) (op 1030): v5

 // Deleting tag 666
 // (op 1260) Deallocate 632
 // Freeing DataFlowTag632: 1 VGPR (Value: UInt32) (op 924): v6

 // Deleting tag 632
 // (op 1260) Deallocate 603
 // Deleting tag 603
 // (op 1260) Deallocate 661
 // Deleting tag 661
 // (op 1260) Deallocate 675
 // Deleting tag 675
 // (op 1260) Deallocate 674
 // Freeing DataFlowTag674: 1 VGPR (Value: UInt32) (op 1054): v10

 // Deleting tag 674
 // (op 1260) Deallocate 677
 // Deleting tag 677
 // (op 1260) Deallocate 619
 // Deleting tag 619
 // (op 1260) Deallocate 667
 // Deleting tag 667
 // (op 1260) Deallocate 651
 // Deleting tag 651
 // (op 1260) Deallocate 616
 // Freeing DataFlowTag616: 1 VGPR (Value: UInt32) (op 871): v0

 // Deleting tag 616
 // (op 1260) Deallocate 617
 // Deleting tag 617
 // (op 1260) Deallocate 659
 // Deleting tag 659
 // (op 1260) Deallocate 600
 // Freeing DataFlowTag600: 1 VGPR (Value: UInt32) (op 818): v4

 // Deleting tag 600
 // (op 1260) Deallocate 601
 // Deleting tag 601
 // (op 1260) Deallocate 653
 // Deleting tag 653
 // (op 1260) Deallocate 669
 // Deleting tag 669
 // (op 1260) Deallocate 585
 // Deleting tag 585
 // (op 1260) Deallocate 587
 // Deleting tag 587
 // (op 1) Deallocate{}(1260) END
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
s_cbranch_vccz RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133 // (op 133) Condition: False, jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133
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
 // (op 348) NOP(934) BEGIN
 // (op 934) generate({})
 // (op 934) end: generate({})
 // (op 348) NOP(934) END
 // (op 348) NOP(931) BEGIN
 // (op 931) generate({})
 // (op 931) end: generate({})
 // (op 348) NOP(931) END
 // (op 348) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(938) BEGIN
 // (op 938) Assign dim(636) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}
 // tag 636: v**UNALLOCATED**
 // (op 938) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32} into DataFlowTag636: VGPR Value: UInt32 x 1: (unallocated)
 // (op 938) reg expression
 // (op 938) reg expression
 // (op 938) reg expression
 // (op 938) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 938) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 938): v0
v_lshrrev_b32 v0, 6, v1 // (op 938) 
 // (op 938) convert
 // (op 938) BEGIN: Tile[1]
 // (op 938) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 938): v10
v_and_b32 v10, 63, v1 // (op 938) 
 // (op 938) END: Tile[1]
 // (op 938) LogicalShiftR({Tile[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 938): v3
v_lshrrev_b32 v3, 5, v10 // (op 938) 
 // (op 938) BEGIN: Flatten
 // (op 938) {Flatten: ShiftL(v3:U32, 2:U32)U32}
 // (op 938) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 938)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 938)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 938): v4
v_lshlrev_b32 v4, 2, v3 // (op 938) 
 // (op 938) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 938): v3

 // (op 938) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 938): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 938) 
 // (op 938) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 938): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 938): v0

 // (op 938) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 938): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 938) 
 // (op 938) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 938): v3

 // (op 938) LogicalShiftR({Tile[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 938): v3
v_lshrrev_b32 v3, 2, v10 // (op 938) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 938): v10

 // (op 938) BEGIN: Tile[1]
 // (op 938) {Tile[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 938): v10
v_and_b32 v10, 7, v3 // (op 938) 
 // (op 938) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 938): v3

 // (op 938) BEGIN: Tile[1]
 // (op 938) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 938): v3
v_and_b32 v3, 3, v1 // (op 938) 
 // (op 938) END: Tile[1]
 // (op 938) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 938): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 938) 
 // (op 938) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 938): v3

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 938): v10

 // (op 938) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 938): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 938) 
 // (op 938) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 938): v4

 // (op 938) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 938): v10
v_mul_lo_u32 v10, v3, s30 // (op 938) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 938): v3

 // (op 938) BEGIN: Join
// Allocated DataFlowTag636: 1 VGPR (Value: UInt32) (op 938): v3
v_add_lshl_u32 v3, v0, v10, 2 // (op 938) 
 // (op 938) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 938): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 938): v0

 // (op 348) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(938) END
 // (op 348) Assign VGPR 32:U64(939) BEGIN
 // (op 939) Assign dim(637) = 32:U64
 // (op 939) Assign stride expression
 // (op 348) Assign VGPR 32:U64(939) END
 // (op 348) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(940) BEGIN
 // (op 940) Assign dim(639) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 639: s**UNALLOCATED**
 // (op 940) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag639: SGPR Buffer: None x 1: (unallocated)
 // (op 940) reg expression
 // (op 940) reg expression
 // (op 940) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 940) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 940)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 940)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 940): s14, s15
s_mul_i32 s15, s30, s13 // (op 940) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 940): s0
s_mul_i32 s0, s31, s12 // (op 940) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 940) least significant
// Allocated : 1 SGPR (Value: Int32) (op 940): s1
s_mul_hi_u32 s1, s30, s12 // (op 940) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 940) most significant: sum
s_add_u32 s15, s15, s1 // (op 940) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 940): s1

 // Freeing : 1 SGPR (Value: Int32) (op 940): s0

 // (op 940) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 940) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 940)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 940)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 940): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 940) 
 // Freeing : 2 SGPRs (Value: Int64) (op 940): s14, s15

 // (op 940) reg expression
 // (op 940) reg expression
// Allocated DataFlowTag639: 4 SGPRs (Buffer: None) (op 940): s8, s9, s10, s11
s_mov_b32 s8, s32 // (op 940) reg expression
s_mov_b32 s9, s33 // (op 940) reg expression
s_mov_b32 s10, s0 // (op 940) reg expression
s_mov_b32 s11, 131072 // (op 940) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 940): s0, s1

 // (op 348) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(940) END
 // (op 348) NOP(932) BEGIN
 // (op 932) generate({})
 // (op 932) end: generate({})
 // (op 348) NOP(932) END
 // (op 348) Assign VGPR 4:U64(945) BEGIN
 // (op 945) Assign dim(641) = 4:U64
 // (op 945) Assign stride expression
 // (op 348) Assign VGPR 4:U64(945) END
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
// VMEM: Expected complete at 5467 (current 4987)
buffer_store_dwordx4 v[32:35], v3, s[8:11], 0 offen offset:32 // (op 56) Store value
// VMEM: Expected complete at 5468 (current 4988)
buffer_store_dwordx4 v[36:39], v3, s[8:11], 0 offen offset:64 // (op 56) Store value
// VMEM: Expected complete at 5469 (current 4989)
buffer_store_dwordx4 v[40:43], v3, s[8:11], 0 offen offset:96 // (op 56) Store value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 5948 (current 5468)
 // (op 348) StoreTiled Value: Float(56) END
 // (op 348) Deallocate{}(1178) BEGIN
 // (op 1178) Deallocate 640
 // Deleting tag 640
 // (op 1178) Deallocate 636
 // Freeing DataFlowTag636: 1 VGPR (Value: UInt32) (op 938): v3

 // Deleting tag 636
 // (op 1178) Deallocate 641
 // Deleting tag 641
 // (op 1178) Deallocate 637
 // Deleting tag 637
 // (op 1178) Deallocate 59
 // Freeing DataFlowTag59: 16 VGPRs (Value: Float) (op 50): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // Deleting tag 59
 // (op 348) Deallocate{}(1178) END
 // (op 348) end: generate({351})
 // (op 348) Unlock Scope 348
 // Unlock None
 // (op 133) Scope(348) END
 // (op 133) end: generate({348})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 133) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 133) 
s_branch RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133 // (op 133) Condition: Done, jump to RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalFalse_0_beta_133:
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
 // (op 354) Scope(702) BEGIN
 // (op 702) Lock Scope 702
 // Lock Branch
 // (op 702) generate({699})
 // (op 702) NOP(699) BEGIN
 // (op 699) generate({})
 // (op 699) end: generate({})
 // (op 702) NOP(699) END
 // (op 702) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(324)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}(707) BEGIN
 // (op 707) Assign dim(566) = {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(324)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}
 // tag 566: v**UNALLOCATED**
 // (op 707) Generate {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(324)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32} into DataFlowTag566: VGPR Value: UInt32 x 1: (unallocated)
 // (op 707) reg expression
 // (op 707) reg expression
 // (op 707) reg expression
 // (op 707) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 707) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 707): v0
v_lshrrev_b32 v0, 6, v1 // (op 707) 
 // (op 707) convert
 // (op 707) BEGIN: Flatten[1]
 // (op 707) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 707): v10
v_and_b32 v10, 63, v1 // (op 707) 
 // (op 707) END: Flatten[1]
 // (op 707) LogicalShiftR({Flatten[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 707): v3
v_lshrrev_b32 v3, 5, v10 // (op 707) 
 // (op 707) BEGIN: Tile
 // (op 707) {Tile: ShiftL(v3:U32, 2:U32)U32}
 // (op 707) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 707)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 707)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 707): v4
v_lshlrev_b32 v4, 2, v3 // (op 707) 
 // (op 707) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 707): v3

 // (op 707) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 707): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 707) 
 // (op 707) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 707): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 707): v0

 // (op 707) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 707): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 707) 
 // (op 707) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 707): v3

 // (op 707) LogicalShiftR({Flatten[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 707): v3
v_lshrrev_b32 v3, 2, v10 // (op 707) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 707): v10

 // (op 707) BEGIN: Flatten[1]
 // (op 707) {Flatten[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 707): v10
v_and_b32 v10, 7, v3 // (op 707) 
 // (op 707) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 707): v3

 // (op 707) BEGIN: Flatten[1]
 // (op 707) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 707): v3
v_and_b32 v3, 3, v1 // (op 707) 
 // (op 707) END: Flatten[1]
 // (op 707) BEGIN: TileTile
// Allocated : 1 VGPR (Value: UInt32) (op 707): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 707) 
 // (op 707) END: TileTile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 707): v3

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 707): v10

 // (op 707) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 707): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 707) 
 // (op 707) END: Tile
 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 707): v4

 // (op 707) Multiply({Tile: v3:U32}, { convertForwardSubsetTensor_10_stride_1_10: s34:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 707): v10
v_mul_lo_u32 v10, v3, s34 // (op 707) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 707): v3

 // (op 707) BEGIN: Split
// Allocated DataFlowTag566: 1 VGPR (Value: UInt32) (op 707): v3
v_add_lshl_u32 v3, v0, v10, 2 // (op 707) 
 // (op 707) END: Split
 // Freeing : 1 VGPR (Value: UInt32) (op 707): v10

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 707): v0

 // (op 702) Assign VGPR {Split: AddShiftL({Tile: ShiftLAdd(DataFlowTag(435)U32, 5:U32, {Tile: ShiftLAdd(LogicalShiftR(DataFlowTag(324)U32, 6:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Tile: ShiftLAdd(DataFlowTag(437)U32, 5:U32, {TileTile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd(DataFlowTag(324)U32, 3:U32)U32})U32})U32}, Convert(Tensor_10_stride_1_10:I64)U32)U32, 2:U32)U32}(707) END
 // (op 702) Deallocate{}(1280) BEGIN
 // (op 1280) Deallocate 435
 // Deleting tag 435
 // (op 1280) Deallocate 437
 // Deleting tag 437
 // (op 1280) Deallocate 324
 // Deleting tag 324
 // (op 702) Deallocate{}(1280) END
 // (op 702) Assign VGPR 32:U64(708) BEGIN
 // (op 708) Assign dim(567) = 32:U64
 // (op 708) Assign stride expression
 // (op 702) Assign VGPR 32:U64(708) END
 // (op 702) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(709) BEGIN
 // (op 709) Assign dim(569) = Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // tag 569: s**UNALLOCATED**
 // (op 709) Generate Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag569: SGPR Buffer: None x 1: (unallocated)
 // (op 709) reg expression
 // (op 709) reg expression
 // (op 709) Multiply({Tensor_10_stride_1_10: s[34:35]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 709) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 709)           lhs (Tensor_10_stride_1_10: SGPR Value: Int64 x 1: s[34:35]) 
 // (op 709)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 709): s14, s15
s_mul_i32 s15, s34, s13 // (op 709) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 709): s0
s_mul_i32 s0, s35, s12 // (op 709) most significant: low of high * low
s_mul_i32 s14, s34, s12 // (op 709) least significant
// Allocated : 1 SGPR (Value: Int32) (op 709): s1
s_mul_hi_u32 s1, s34, s12 // (op 709) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 709) most significant: sum
s_add_u32 s15, s15, s1 // (op 709) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 709): s1

 // Freeing : 1 SGPR (Value: Int32) (op 709): s0

 // (op 709) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 709) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 709)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 709)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 709): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 709) 
 // Freeing : 2 SGPRs (Value: Int64) (op 709): s14, s15

 // (op 709) reg expression
 // (op 709) reg expression
// Allocated DataFlowTag569: 4 SGPRs (Buffer: None) (op 709): s4, s5, s6, s7
s_mov_b32 s4, s36 // (op 709) reg expression
s_mov_b32 s5, s37 // (op 709) reg expression
s_mov_b32 s6, s0 // (op 709) reg expression
s_mov_b32 s7, 131072 // (op 709) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 709): s0, s1

 // (op 702) Assign SGPR Concatenate(Tensor_10_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_10_stride_1_10:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(709) END
 // (op 702) Deallocate{Tensor_10_pointer}(1355) BEGIN
 // (op 1355) Deallocate Tensor_10_pointer
 // Freeing Tensor_10_pointer: 2 SGPRs (Value: Raw32): s36, s37

 // (op 702) Deallocate{Tensor_10_pointer}(1355) END
 // (op 702) NOP(700) BEGIN
 // (op 700) generate({})
 // (op 700) end: generate({})
 // (op 702) NOP(700) END
 // (op 702) Assign VGPR 4:U64(714) BEGIN
 // (op 714) Assign dim(571) = 4:U64
 // (op 714) Assign stride expression
 // (op 702) Assign VGPR 4:U64(714) END
 // (op 702) LoadTiled Value: Float(145) BEGIN
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
// VMEM: Expected complete at 6456 (current 5976)
buffer_load_dwordx4 v[16:19], v3, s[4:7], 0 offen offset:32 // (op 145) Load value
// VMEM: Expected complete at 6457 (current 5977)
buffer_load_dwordx4 v[20:23], v3, s[4:7], 0 offen offset:64 // (op 145) Load value
// VMEM: Expected complete at 6458 (current 5978)
buffer_load_dwordx4 v[24:27], v3, s[4:7], 0 offen offset:96 // (op 145) Load value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 6937 (current 6457)
 // (op 702) LoadTiled Value: Float(145) END
 // (op 702) Deallocate{Tensor_10_stride_1_10}(1190) BEGIN
 // (op 1190) Deallocate 5
 // Deleting tag 5
 // (op 1190) Deallocate 566
 // Freeing DataFlowTag566: 1 VGPR (Value: UInt32) (op 707): v3

 // Deleting tag 566
 // (op 1190) Deallocate 570
 // Deleting tag 570
 // (op 1190) Deallocate 571
 // Deleting tag 571
 // (op 1190) Deallocate 569
 // Freeing DataFlowTag569: 4 SGPRs (Buffer: None) (op 709): s4, s5, s6, s7

 // Deleting tag 569
 // (op 1190) Deallocate 567
 // Deleting tag 567
 // (op 1190) Deallocate Tensor_10_stride_1_10
 // Freeing Tensor_10_stride_1_10: 2 SGPRs (Value: Raw32): s34, s35

 // (op 702) Deallocate{Tensor_10_stride_1_10}(1190) END
 // (op 702) end: generate({699})
 // (op 702) Unlock Scope 702
 // Unlock None
 // (op 354) Scope(702) END
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
 // (op 354) Deallocate{}(1192) BEGIN
 // (op 1192) Deallocate 419
 // Freeing : 16 VGPRs (Value: Float) (op 145): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // Deleting tag 419
 // (op 354) Deallocate{}(1192) END
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
 // (op 354) Deallocate{}(1186) BEGIN
 // (op 1186) Deallocate 423
 // Freeing DataFlowTag423: 16 VGPRs (Value: Float) (op 142): v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56, v57, v58, v59

 // Deleting tag 423
 // (op 1186) Deallocate 421
 // Freeing DataFlowTag421: 16 VGPRs (Value: Float) (op 146): v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43

 // Deleting tag 421
 // (op 354) Deallocate{}(1186) END
 // (op 354) NOP(951) BEGIN
 // (op 951) generate({})
 // (op 951) end: generate({})
 // (op 354) NOP(951) END
 // (op 354) NOP(948) BEGIN
 // (op 948) generate({})
 // (op 948) end: generate({})
 // (op 354) NOP(948) END
 // (op 354) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(955) BEGIN
 // (op 955) Assign dim(642) = {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}
 // tag 642: v**UNALLOCATED**
 // (op 955) Generate {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32} into DataFlowTag642: VGPR Value: UInt32 x 1: (unallocated)
 // (op 955) reg expression
 // (op 955) reg expression
 // (op 955) reg expression
 // (op 955) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 955) reg expression
// Allocated : 1 VGPR (Value: UInt32) (op 955): v0
v_lshrrev_b32 v0, 6, v1 // (op 955) 
 // (op 955) convert
 // (op 955) BEGIN: Tile[1]
 // (op 955) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 955): v10
v_and_b32 v10, 63, v1 // (op 955) 
 // (op 955) END: Tile[1]
 // (op 955) LogicalShiftR({Tile[1]: v10:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 955): v3
v_lshrrev_b32 v3, 5, v10 // (op 955) 
 // (op 955) BEGIN: Flatten
 // (op 955) {Flatten: ShiftL(v3:U32, 2:U32)U32}
 // (op 955) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 955)         value (VGPR Value: UInt32 x 1: v3) 
 // (op 955)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 955): v4
v_lshlrev_b32 v4, 2, v3 // (op 955) 
 // (op 955) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 955): v3

 // (op 955) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 955): v3
v_lshl_add_u32 v3, v0, 5, v4 // (op 955) 
 // (op 955) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 955): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 955): v0

 // (op 955) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 955): v0
v_lshl_add_u32 v0, s16, 5, v3 // (op 955) 
 // (op 955) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 955): v3

 // (op 955) LogicalShiftR({Tile[1]: v10:U32}, 2:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 955): v3
v_lshrrev_b32 v3, 2, v10 // (op 955) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 955): v10

 // (op 955) BEGIN: Tile[1]
 // (op 955) {Tile[1]: BitwiseAnd(v3:U32, 7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 955): v10
v_and_b32 v10, 7, v3 // (op 955) 
 // (op 955) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 955): v3

 // (op 955) BEGIN: Tile[1]
 // (op 955) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 955): v3
v_and_b32 v3, 3, v1 // (op 955) 
 // (op 955) END: Tile[1]
 // (op 955) BEGIN: FlattenFlatten
// Allocated : 1 VGPR (Value: UInt32) (op 955): v4
v_lshl_add_u32 v4, v10, 2, v3 // (op 955) 
 // (op 955) END: FlattenFlatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 955): v3

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 955): v10

 // (op 955) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 955): v3
v_lshl_add_u32 v3, s17, 5, v4 // (op 955) 
 // (op 955) END: Flatten
 // Freeing FlattenFlatten: 1 VGPR (Value: UInt32) (op 955): v4

 // (op 955) Multiply({Flatten: v3:U32}, { convertForwardSubsetTensor_21_stride_1_16: s30:U32})U32
// Allocated : 1 VGPR (Value: UInt32) (op 955): v10
v_mul_lo_u32 v10, v3, s30 // (op 955) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 955): v3

 // (op 955) BEGIN: Join
// Allocated DataFlowTag642: 1 VGPR (Value: UInt32) (op 955): v3
v_add_lshl_u32 v3, v0, v10, 2 // (op 955) 
 // (op 955) END: Join
 // Freeing : 1 VGPR (Value: UInt32) (op 955): v10

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 955): v0

 // (op 354) Assign VGPR {Join: AddShiftL({Flatten: ShiftLAdd(DataFlowTag(439)U32, 5:U32, {Flatten: ShiftLAdd(LogicalShiftR(DataFlowTag(364)U32, 6:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, Multiply({Flatten: ShiftLAdd(DataFlowTag(441)U32, 5:U32, {FlattenFlatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd(DataFlowTag(364)U32, 3:U32)U32})U32})U32}, Convert(Tensor_21_stride_1_16:I64)U32)U32, 2:U32)U32}(955) END
 // (op 354) Assign VGPR 32:U64(956) BEGIN
 // (op 956) Assign dim(643) = 32:U64
 // (op 956) Assign stride expression
 // (op 354) Assign VGPR 32:U64(956) END
 // (op 354) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(957) BEGIN
 // (op 957) Assign dim(639) = Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB
 // (op 957) Generate Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB into DataFlowTag639: SGPR Buffer: None x 1: s[8:11]
 // (op 957) reg expression
 // (op 957) reg expression
 // (op 957) Multiply({Tensor_21_stride_1_16: s[30:31]:I64}, {Tensor_2_size_1_4: s[12:13]:I64})I64
 // (op 957) Multiply: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 957)           lhs (Tensor_21_stride_1_16: SGPR Value: Int64 x 1: s[30:31]) 
 // (op 957)           rhs (Tensor_2_size_1_4: SGPR Value: Int64 x 1: s[12:13])
// Allocated : 2 SGPRs (Value: Int64) (op 957): s14, s15
s_mul_i32 s15, s30, s13 // (op 957) most significant: low of low * high
// Allocated : 1 SGPR (Value: Int32) (op 957): s0
s_mul_i32 s0, s31, s12 // (op 957) most significant: low of high * low
s_mul_i32 s14, s30, s12 // (op 957) least significant
// Allocated : 1 SGPR (Value: Int32) (op 957): s1
s_mul_hi_u32 s1, s30, s12 // (op 957) most significant: high of low * low
s_add_u32 s15, s15, s0 // (op 957) most significant: sum
s_add_u32 s15, s15, s1 // (op 957) most significant: sum
 // Freeing : 1 SGPR (Value: Int32) (op 957): s1

 // Freeing : 1 SGPR (Value: Int32) (op 957): s0

 // (op 957) ShiftL(s[14:15]:I64, 2:U32)I64
 // (op 957) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 957)         value (SGPR Value: Int64 x 1: s[14:15]) 
 // (op 957)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 957): s0, s1
s_lshl_b64 s[0:1], s[14:15], 2 // (op 957) 
 // Freeing : 2 SGPRs (Value: Int64) (op 957): s14, s15

 // (op 957) reg expression
 // (op 957) reg expression
s_mov_b32 s8, s32 // (op 957) reg expression
s_mov_b32 s9, s33 // (op 957) reg expression
s_mov_b32 s10, s0 // (op 957) reg expression
s_mov_b32 s11, 131072 // (op 957) call()
 // Freeing : 2 SGPRs (Value: Int64) (op 957): s0, s1

 // (op 354) Assign SGPR Concatenate(Tensor_21_pointer:PG, BitFieldExtract(ShiftL(Multiply(Tensor_21_stride_1_16:I64, Tensor_2_size_1_4:I64)I64, 2:U32)I64, width:32, offset:0)R, 131072:R)PB(957) END
 // (op 354) NOP(949) BEGIN
 // (op 949) generate({})
 // (op 949) end: generate({})
 // (op 354) NOP(949) END
 // (op 354) Assign VGPR 4:U64(962) BEGIN
 // (op 962) Assign dim(645) = 4:U64
 // (op 962) Assign stride expression
 // (op 354) Assign VGPR 4:U64(962) END
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
// VMEM: Expected complete at 7463 (current 6983)
buffer_store_dwordx4 v[16:19], v3, s[8:11], 0 offen offset:32 // (op 144) Store value
// VMEM: Expected complete at 7464 (current 6984)
buffer_store_dwordx4 v[20:23], v3, s[8:11], 0 offen offset:64 // (op 144) Store value
// VMEM: Expected complete at 7465 (current 6985)
buffer_store_dwordx4 v[24:27], v3, s[8:11], 0 offen offset:96 // (op 144) Store value
// VMEM: Expected stall of 478. CBNW: 0, Inc: 3
// VMEM: Expected complete at 7944 (current 7464)
 // (op 354) StoreTiled Value: Float(144) END
 // (op 354) Deallocate{}(1188) BEGIN
 // (op 1188) Deallocate 644
 // Deleting tag 644
 // (op 1188) Deallocate 645
 // Deleting tag 645
 // (op 1188) Deallocate 643
 // Deleting tag 643
 // (op 1188) Deallocate 642
 // Freeing DataFlowTag642: 1 VGPR (Value: UInt32) (op 955): v3

 // Deleting tag 642
 // (op 1188) Deallocate 425
 // Freeing DataFlowTag425: 16 VGPRs (Value: Float) (op 143): v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27

 // Deleting tag 425
 // (op 354) Deallocate{}(1188) END
 // (op 354) end: generate({357})
 // (op 354) Unlock Scope 354
 // Unlock None
 // (op 133) Scope(354) END
 // (op 133) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1262) BEGIN
 // (op 1262) Deallocate 639
 // Freeing DataFlowTag639: 4 SGPRs (Buffer: None) (op 940): s8, s9, s10, s11

 // Deleting tag 639
 // (op 1262) Deallocate 441
 // Deleting tag 441
 // (op 1262) Deallocate 439
 // Deleting tag 439
 // (op 1262) Deallocate 364
 // Deleting tag 364
 // (op 1262) Deallocate Tensor_21_pointer
 // Freeing Tensor_21_pointer: 2 SGPRs (Value: Raw32): s32, s33

 // (op 1262) Deallocate Tensor_21_stride_1_16
 // Freeing Tensor_21_stride_1_16: 2 SGPRs (Value: Raw32): s30, s31

 // (op 133) Deallocate{Tensor_21_pointer, Tensor_21_stride_1_16}(1262) END
 // (op 133) end: generate({354})
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// REMOVEME: Wait before conditional label!
 // (op 133) 
RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_ConditionalBottom_0_beta_133:
 // (op 133) 
 // (op 133) Unlock Conditional
 // Unlock None
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(44)S)BL(133) END
 // (op 1) Deallocate{Tensor_2_size_1_4}(1184) BEGIN
 // (op 1184) Deallocate 55
 // Freeing DataFlowTag55: 16 ACCVGPRs (Value: Float) (op 63): a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15

 // Deleting tag 55
 // (op 1184) Deallocate 53
 // Freeing : 1 VGPR (Value: Float) (op 43): v11

 // Deleting tag 53
 // (op 1184) Deallocate 44
 // Freeing : 1 VGPR (Value: Float) (op 2): v9

 // Deleting tag 44
 // (op 1184) Deallocate Tensor_2_size_1_4
 // Freeing Tensor_2_size_1_4: 2 SGPRs (Value: Raw32): s12, s13

 // (op 1) Deallocate{Tensor_2_size_1_4}(1184) END
 // (op 1) end: generate({761, 814, 867, 920, 978, 1002, 1026, 1050})
 // Kernel(1) END
 // end: generate({1})
 // CodeGeneratorVisitor::generate() end
s_endpgm  // End of RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.LRR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end:
.size RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2, .LRR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2_end-RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
.rodata
.p2align 6
.amdhsa_kernel RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
 // Resource limits
  .amdhsa_next_free_vgpr 88
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 6400 // lds bytes
  .amdhsa_accum_offset 72
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
    DataTypeB: F4
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
  - .name: RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2
    .symbol: RR_GEMM_TN_BF8_FP4_Float_Float_Float_SA_BE8M0_32_SB_BE8M0_32_WGT_32x32x64_UR_2.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 6400
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 56
    .vgpr_count: 72
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
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_2_pointer, variableType: {dataType: FP4, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: FP4, pointerType: PointerGlobal}
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

