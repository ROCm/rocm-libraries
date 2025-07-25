
 // FastArithmetic:	orig = CommandArgument(Tensor_0_extent)I64
 // 	x = Tensor_0_extent_0:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_2_extent)I64
 // 	x = Tensor_2_extent_2:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_4_extent)I64
 // 	x = Tensor_4_extent_4:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_0_size_0)I64
 // 	x = Tensor_0_size_0_8:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_0_size_1)I64
 // 	x = Tensor_0_size_1_9:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_0_stride_1)I64
 // 	x = Tensor_0_stride_1_10:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_0_size_1)I64
 // 	x = Tensor_0_size_1_9:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_2_stride_0)I64
 // 	x = Tensor_2_stride_0_11:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_2_size_1)I64
 // 	x = Tensor_2_size_1_12:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_0_size_0)I64
 // 	x = Tensor_0_size_0_8:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_2_size_1)I64
 // 	x = Tensor_2_size_1_12:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_4_stride_1)I64
 // 	x = Tensor_4_stride_1_13:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_0_size_0)I64
 // 	x = Tensor_0_size_0_8:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_2_size_1)I64
 // 	x = Tensor_2_size_1_12:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_15_stride_1)I64
 // 	x = Tensor_15_stride_1_14:I64

 // FastArithmetic:	orig = CommandArgument(Tensor_15_extent)I64
 // 	x = Tensor_15_extent_15:I64

 // FastArithmetic:	orig = Divide(Subtract(Add(CommandArgument(Tensor_0_size_0)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64
 // 	x = ArithmeticShiftR(Subtract(Add(Tensor_0_size_0_8:I64, 64:U32)I64, 1:U32)I64, 6:U32)I64

 // FastArithmetic:	orig = Divide(Subtract(Add(CommandArgument(Tensor_0_size_1)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64
 // 	x = ArithmeticShiftR(Subtract(Add(Tensor_0_size_1_9:I64, 64:U32)I64, 1:U32)I64, 6:U32)I64

 // FastArithmetic:	orig = Divide(Subtract(Add(CommandArgument(Tensor_0_size_1)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64
 // 	x = ArithmeticShiftR(Subtract(Add(Tensor_0_size_1_9:I64, 64:U32)I64, 1:U32)I64, 6:U32)I64

 // FastArithmetic:	orig = Divide(Subtract(Add(CommandArgument(Tensor_2_size_1)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64
 // 	x = ArithmeticShiftR(Subtract(Add(Tensor_2_size_1_12:I64, 64:U32)I64, 1:U32)I64, 6:U32)I64

 // FastArithmetic:	orig = Divide(Subtract(Add(CommandArgument(Tensor_0_size_0)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64
 // 	x = ArithmeticShiftR(Subtract(Add(Tensor_0_size_0_8:I64, 64:U32)I64, 1:U32)I64, 6:U32)I64

 // FastArithmetic:	orig = Divide(Subtract(Add(CommandArgument(Tensor_2_size_1)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64
 // 	x = ArithmeticShiftR(Subtract(Add(Tensor_2_size_1_12:I64, 64:U32)I64, 1:U32)I64, 6:U32)I64

 // FastArithmetic:	orig = Convert(Divide(CommandArgument(Tensor_0_size_1)I64, 64:U32)I64)U32
 // 	x = Convert(ArithmeticShiftR(Tensor_0_size_1_9:I64, 6:U32)I64)U32

 // FastArithmetic:	orig = Convert(Divide(CommandArgument(Tensor_0_size_1)I64, 64:U32)I64)U32
 // 	x = Convert(ArithmeticShiftR(Tensor_0_size_1_9:I64, 6:U32)I64)U32

 // FastArithmetic:	orig = Convert(Divide(Subtract(Add(CommandArgument(Tensor_0_size_0)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64)I
 // 	x = Convert(ArithmeticShiftR(Subtract(Add(Tensor_0_size_0_8:I64, 64:U32)I64, 1:U32)I64, 6:U32)I64)I

 // FastArithmetic:	orig = Convert(Divide(Subtract(Add(CommandArgument(Tensor_0_size_0)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64)I
 // 	x = Convert(ArithmeticShiftR(Subtract(Add(Tensor_0_size_0_8:I64, 64:U32)I64, 1:U32)I64, 6:U32)I64)I

 // FastArithmetic:	orig = Convert(Divide(Subtract(Add(CommandArgument(Tensor_0_size_0)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64)I
 // 	x = Convert(ArithmeticShiftR(Subtract(Add(Tensor_0_size_0_8:I64, 64:U32)I64, 1:U32)I64, 6:U32)I64)I

 // FastArithmetic:	orig = Convert(Divide(Subtract(Add(CommandArgument(Tensor_2_size_1)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64)I
 // 	x = Convert(ArithmeticShiftR(Subtract(Add(Tensor_2_size_1_12:I64, 64:U32)I64, 1:U32)I64, 6:U32)I64)I

 // FastArithmetic:	orig = Convert(Divide(Subtract(Add(CommandArgument(Tensor_2_size_1)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64)I
 // 	x = Convert(ArithmeticShiftR(Subtract(Add(Tensor_2_size_1_12:I64, 64:U32)I64, 1:U32)I64, 6:U32)I64)I

 // FastArithmetic:	orig = Convert(Divide(Subtract(Add(CommandArgument(Tensor_2_size_1)I64, 64:U32)I64, 1:U32)I64, 64:U32)I64)I
 // 	x = Convert(ArithmeticShiftR(Subtract(Add(Tensor_2_size_1_12:I64, 64:U32)I64, 1:U32)I64, 6:U32)I64)I

 // FastArithmetic:	orig = LessThan(DataFlowTag(277)U32, Convert(Divide(CommandArgument(Tensor_0_size_1)I64, 64:U32)I64)U32)BL
 // 	x = LessThan(DataFlowTag(277)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_9:I64, 6:U32)I64)U32)BL

 // FastArithmetic:	orig = Convert(Add(DataFlowTag(443)U32, Multiply(1:U32, DataFlowTag(444)U64)U64)U64)U32
 // 	x = Convert(Add(DataFlowTag(443)U32, DataFlowTag(444)U64)U64)U32

 // FastArithmetic:	orig = Convert(Add(DataFlowTag(451)U32, Multiply(1:U32, DataFlowTag(452)U64)U64)U64)U32
 // 	x = Convert(Add(DataFlowTag(451)U32, DataFlowTag(452)U64)U64)U32

 // Tensor.Float.d2 0, (base=&0, lim=&8, sizes={&16 &24 }, strides={&32 &40 })
 // T_LOAD_TILED 1 Source 0
 // Tensor.Float.d2 2, (base=&48, lim=&56, sizes={&64 &72 }, strides={&80 &88 })
 // T_LOAD_TILED 3 Source 2
 // Tensor.Float.d2 4, (base=&96, lim=&104, sizes={&112 &120 }, strides={&128 &136 })
 // T_LOAD_TILED 5 Source 4
 // Scalar.Float.6
 // T_LOAD_SCALAR 7Source 6
 // Scalar.Float.8
 // T_LOAD_SCALAR 9Source 8
 // T_Mul 1 3 Value: Float
 // T_EXECUTE 7 10 9 5
 //   E_Mul 11, 9, 5
 //   E_Mul 12, 7, 10
 //   E_Add 13, 11, 12
 // Tensor.Float.d2 15, (base=&152, lim=&160, sizes={&168 &176 }, strides={&184 &192 })
 // T_STORE_TILED 16 Source 13 Dest 15

 // Tensor_0_pointer: PointerGlobal: Float(offset: 0, size: 8, read_write)
 // Tensor_0_extent: Value: Int64(offset: 8, size: 8, read_only)
 // Tensor_0_size_0: Value: Int64(offset: 16, size: 8, read_only)
 // Tensor_0_size_1: Value: Int64(offset: 24, size: 8, read_only)
 // Tensor_0_stride_0: Value: Int64(offset: 32, size: 8, read_only)
 // Tensor_0_stride_1: Value: Int64(offset: 40, size: 8, read_only)
 // Tensor_2_pointer: PointerGlobal: Float(offset: 48, size: 8, read_write)
 // Tensor_2_extent: Value: Int64(offset: 56, size: 8, read_only)
 // Tensor_2_size_0: Value: Int64(offset: 64, size: 8, read_only)
 // Tensor_2_size_1: Value: Int64(offset: 72, size: 8, read_only)
 // Tensor_2_stride_0: Value: Int64(offset: 80, size: 8, read_only)
 // Tensor_2_stride_1: Value: Int64(offset: 88, size: 8, read_only)
 // Tensor_4_pointer: PointerGlobal: Float(offset: 96, size: 8, read_write)
 // Tensor_4_extent: Value: Int64(offset: 104, size: 8, read_only)
 // Tensor_4_size_0: Value: Int64(offset: 112, size: 8, read_only)
 // Tensor_4_size_1: Value: Int64(offset: 120, size: 8, read_only)
 // Tensor_4_stride_0: Value: Int64(offset: 128, size: 8, read_only)
 // Tensor_4_stride_1: Value: Int64(offset: 136, size: 8, read_only)
 // user_Float_Value_6: Value: Float(offset: 144, size: 4, read_only)
 // user_Float_Value_8: Value: Float(offset: 148, size: 4, read_only)
 // Tensor_15_pointer: PointerGlobal: Float(offset: 152, size: 8, read_write)
 // Tensor_15_extent: Value: Int64(offset: 160, size: 8, read_only)
 // Tensor_15_size_0: Value: Int64(offset: 168, size: 8, read_only)
 // Tensor_15_size_1: Value: Int64(offset: 176, size: 8, read_only)
 // Tensor_15_stride_0: Value: Int64(offset: 184, size: 8, read_only)
 // Tensor_15_stride_1: Value: Int64(offset: 192, size: 8, read_only)
 // SCRATCH: PointerGlobal: UInt32(offset: 200, size: 8, read_write)

.amdgcn_target "amdgcn-amd-amdhsa--gfx942:sramecc+"
.set .amdgcn.next_free_vgpr, 0
.set .amdgcn.next_free_sgpr, 0
.text
.globl GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel
.p2align 8
.type GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel,@function
// Allocated Kernel argument pointer: 2 SGPRs (PointerGlobal: Raw32): s0, s1
// Allocated Workgroup Index X: 1 SGPR (Value: UInt32): s2
// Allocated Workgroup Index Y: 1 SGPR (Value: UInt32): s3
// Allocated Packed Workitem Index: 1 VGPR (Value: Raw32): v0
// Allocated Workitem Index X: 1 VGPR (Value: UInt32): v1
// Allocated Workitem Index Y: 1 VGPR (Value: UInt32): v2
GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel:

 // Kernel Options:
 //   logLevel:                             Verbose
 //   alwaysWaitAfterLoad:                    false
 //   alwaysWaitAfterStore:                   false
 //   alwaysWaitBeforeBranch:                 false
 //   alwaysWaitZeroBeforeBarrier:            false
 //   preloadKernelArguments:                  true
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
 //   lazyAddArguments:                        true
 //   minLaunchTimeExpressionComplexity:         10
 //   maxConcurrentSubExpressions:                2
 //   maxConcurrentControlOps:                 none
 //   enableFullDivision:                     false

 // Loading Kernel Arguments: 
 // KernelArg{Tensor_0_extent_0, Value: Int64, CommandArgument(Tensor_0_extent)I64, o:0, s:8}
 // KernelArg{Tensor_0_pointer, PointerGlobal: Float, read_write, CommandArgument(Tensor_0_pointer)PG, o:8, s:8}
 // KernelArg{Tensor_2_extent_2, Value: Int64, CommandArgument(Tensor_2_extent)I64, o:16, s:8}
 // KernelArg{Tensor_2_pointer, PointerGlobal: Float, read_write, CommandArgument(Tensor_2_pointer)PG, o:24, s:8}
 // KernelArg{Tensor_4_extent_4, Value: Int64, CommandArgument(Tensor_4_extent)I64, o:32, s:8}
 // KernelArg{Tensor_4_pointer, PointerGlobal: Float, read_write, CommandArgument(Tensor_4_pointer)PG, o:40, s:8}
 // KernelArg{user_Float_Value_6, Value: Float, CommandArgument(user_Float_Value_6)S, o:48, s:4}
 // KernelArg{user_Float_Value_8, Value: Float, CommandArgument(user_Float_Value_8)S, o:52, s:4}
 // KernelArg{Tensor_0_size_0_8, Value: Int64, CommandArgument(Tensor_0_size_0)I64, o:56, s:8}
 // KernelArg{Tensor_0_size_1_9, Value: Int64, CommandArgument(Tensor_0_size_1)I64, o:64, s:8}
 // KernelArg{Tensor_0_stride_1_10, Value: Int64, CommandArgument(Tensor_0_stride_1)I64, o:72, s:8}
 // KernelArg{Tensor_2_stride_0_11, Value: Int64, CommandArgument(Tensor_2_stride_0)I64, o:80, s:8}
 // KernelArg{Tensor_2_size_1_12, Value: Int64, CommandArgument(Tensor_2_size_1)I64, o:88, s:8}
 // KernelArg{Tensor_4_stride_1_13, Value: Int64, CommandArgument(Tensor_4_stride_1)I64, o:96, s:8}
 // KernelArg{Tensor_15_stride_1_14, Value: Int64, CommandArgument(Tensor_15_stride_1)I64, o:104, s:8}
 // KernelArg{Tensor_15_extent_15, Value: Int64, CommandArgument(Tensor_15_extent)I64, o:112, s:8}
 // KernelArg{Tensor_15_pointer, PointerGlobal: Float, read_write, CommandArgument(Tensor_15_pointer)PG, o:120, s:8}

s_load_dwordx2 s[4:5], s[0:1], 0 // Load scalar value
s_load_dwordx2 s[6:7], s[0:1], 8 // Load scalar value
s_load_dwordx2 s[8:9], s[0:1], 16 // Load scalar value
s_load_dwordx2 s[10:11], s[0:1], 24 // Load scalar value
s_load_dwordx2 s[12:13], s[0:1], 32 // Load scalar value
s_load_dwordx2 s[14:15], s[0:1], 40 // Load scalar value
s_load_dword s16, s[0:1], 48 // Load scalar value
s_load_dword s17, s[0:1], 52 // Load scalar value
s_load_dwordx2 s[18:19], s[0:1], 56 // Load scalar value
s_load_dwordx2 s[20:21], s[0:1], 64 // Load scalar value
s_load_dwordx2 s[22:23], s[0:1], 72 // Load scalar value
s_load_dwordx2 s[24:25], s[0:1], 80 // Load scalar value
s_load_dwordx2 s[26:27], s[0:1], 88 // Load scalar value
s_load_dwordx2 s[28:29], s[0:1], 96 // Load scalar value
s_load_dwordx2 s[30:31], s[0:1], 104 // Load scalar value
s_load_dwordx2 s[32:33], s[0:1], 112 // Load scalar value
s_load_dwordx2 s[34:35], s[0:1], 120 // Load scalar value
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

 // Marking Workgroup Index X: SGPR Value: UInt32 x 1: s2 as read-only
 // Marking Workgroup Index Y: SGPR Value: UInt32 x 1: s3 as read-only
 // Marking Workitem Index X: VGPR Value: UInt32 x 1: v1 as read-only
 // Marking Workitem Index Y: VGPR Value: UInt32 x 1: v2 as read-only
 // CodeGeneratorVisitor::generate() begin
 // IdentifyParallelDimensions
 // OrderMemory
 // UpdateParameters
 // AddLDS
 // LowerLinear
 // LowerTile
 // LowerTensorContraction
 // Simplify
 // ConstantPropagation
 // FuseExpressions
 // ConnectWorkgroups
 // UnrollLoops
 // FuseLoops
 // RemoveDuplicates
 // OrderEpilogueBlocks
 // CleanLoops
 // SwizzleScale
 // AddPrefetch
 // AddF6LDSPadding
 // AddDirect2LDS
 // AddComputeIndex
 // AddPRNG
 // UpdateWavefrontParameters
 // LoadPacked
 // AddConvert
 // NopExtraScopes
 // AddDeallocateDataFlow
 // InlineIncrements
 // InlineInits
 // OrderMultiplyNodes
 // Simplify
 // AliasDataFlowTags
 // CleanArguments
 // AddDeallocateArguments
 // MergeAdjacentDeallocates
 // Simplify
 // SetWorkitemCount
 // generate({1})
 // Kernel(1) BEGIN
 // (op 1) generate({})
 // (op 1) end: generate({})
 // (op 1) generate({593, 602})
 // (op 1) ComputeIndex(593) BEGIN
 // (op 593) KernelGraph::LoadStoreTileGenerator::ComputeIndex(593): target 14 increment 90 base -1 offset 469 stride 470 buffer -1
 // tag 469: v**UNALLOCATED**
 // FastArithmetic:	orig = {Tile: Add(Multiply({Tile: Add(Multiply(0:U32, 2:U32)U32, {Tile: Add(Multiply({Flatten[0]: Divide({Flatten[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 32:U32)U32}, 1:U32)U32, 0:U32)U32})U32}, 64:U32)U32, {Tile: Add(Multiply({Tile: Divide({Flatten[0]: Divide({Workitem Index X: v1:U32}, 64:U32)U32}, 2:U32)U32}, 32:U32)U32, {Flatten[1]: Modulo({Flatten[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 32:U32)U32})U32})U32}
 // 	x = {Tile: ShiftLAdd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 6:U32, {Tile: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32})U32})U32}

 // (op 593)   Offset(469): indexExpr: {Tile: ShiftLAdd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 6:U32, {Tile: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32})U32})U32}
 // (op 593)   Offset(469): paddingBytes: 0:U32
 // (op 593) Generate Convert(Add(Multiply({Tile: ShiftLAdd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 6:U32, {Tile: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32})U32})U32}, 4:U32)U32, 0:U32)U32)U32 into Offset593: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Tile: ShiftLAdd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 6:U32, {Tile: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32})U32})U32}, 4:U32)U32, 0:U32)U32)U32
 // 	x = ShiftL({Tile: ShiftLAdd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 6:U32, {Tile: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32})U32})U32}, 2:U32)U32

 // (op 593) reg expression
 // (op 593) BEGIN: Flatten[1]
 // (op 593) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 593) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 1) ComputeIndex(602) BEGIN
 // (op 602) KernelGraph::LoadStoreTileGenerator::ComputeIndex(602): target 22 increment 149 base -1 offset 477 stride 478 buffer -1
 // tag 477: v**UNALLOCATED**
 // FastArithmetic:	orig = {Tile: Add(Multiply({Tile: Add(Multiply(0:U32, 2:U32)U32, {Tile: Add(Multiply({Flatten[0]: Divide({Flatten[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 32:U32)U32}, 1:U32)U32, 0:U32)U32})U32}, 64:U32)U32, {Tile: Add(Multiply({Tile: Modulo({Flatten[0]: Divide({Workitem Index X: v1:U32}, 64:U32)U32}, 2:U32)U32}, 32:U32)U32, {Flatten[1]: Modulo({Flatten[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 32:U32)U32})U32})U32}
 // 	x = {Tile: ShiftLAdd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 6:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32})U32})U32}

 // (op 602)   Offset(477): indexExpr: {Tile: ShiftLAdd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 6:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32})U32})U32}
 // (op 602)   Offset(477): paddingBytes: 0:U32
 // (op 602) Generate Convert(Add(Multiply({Tile: ShiftLAdd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 6:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32})U32})U32}, 4:U32)U32, 0:U32)U32)U32 into Offset602: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Tile: ShiftLAdd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 6:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32})U32})U32}, 4:U32)U32, 0:U32)U32)U32
 // 	x = ShiftL({Tile: ShiftLAdd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 6:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32})U32})U32}, 2:U32)U32

 // (op 602) reg expression
 // (op 602) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
 // (op 602) BEGIN: Flatten[1]
 // (op 602) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 593): v0
v_and_b32 v0, 63, v1 // (op 593) 
 // (op 593) END: Flatten[1]
// Allocated : 1 VGPR (Value: UInt32) (op 593): v3
v_lshrrev_b32 v3, 7, v1 // (op 593) 
 // (op 593) BEGIN: Flatten[1]
 // (op 593) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
 // (op 593) LogicalShiftR({Flatten[1]: v0:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 593): v4
v_and_b32 v4, 31, v1 // (op 593) 
 // (op 593) END: Flatten[1]
// Allocated : 1 VGPR (Value: UInt32) (op 593): v5
v_lshrrev_b32 v5, 5, v0 // (op 593) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 593): v0

 // (op 593) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 593): v0
v_lshl_add_u32 v0, v3, 5, v4 // (op 593) 
 // (op 593) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 593): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 593): v3

 // (op 593) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 593): v4
v_lshl_add_u32 v4, v5, 6, v0 // (op 593) 
 // (op 593) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 593): v0

 // Freeing : 1 VGPR (Value: UInt32) (op 593): v5

 // (op 593) ShiftL({Tile: v4:U32}, 2:U32)U32
 // (op 593) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 593)         value (Tile: VGPR Value: UInt32 x 1: v4) 
 // (op 593)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated Offset593: 1 VGPR (Value: UInt32) (op 593): v0
v_lshlrev_b32 v0, 2, v4 // (op 593) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 593): v4

 // FastArithmetic:	orig = Add(Multiply(2:U32, 64:U32)U32, 0:U32)U32
 // 	x = 128:U32

 // (op 593)   Stride(470): indexExpr: 128:U32
 // (op 593)   Stride(470): indexExprPaddingBytes: 0:U32
 // (op 593)   Stride(470): unitStride: false vgprBlockSize: 0
 // (op 593)   Stride(470): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 593)   Stride(470): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(128:U32, 4:U32)U32, 0:U32)U32
 // 	x = 512:U32

 // (op 1) ComputeIndex(593) END
// Allocated : 1 VGPR (Value: UInt32) (op 602): v3
v_lshrrev_b32 v3, 6, v1 // (op 602) 
 // (op 1) ComputeIndex(594) BEGIN
 // (op 594) KernelGraph::LoadStoreTileGenerator::ComputeIndex(594): target 14 increment 99 base 469 offset 471 stride 472 buffer -1
 // FastArithmetic:	orig = Add(Multiply(1:U32, 64:U32)U32, 0:U32)U32
 // 	x = 64:U32

 // (op 594)   Stride(472): indexExpr: 64:U32
 // (op 594)   Stride(472): indexExprPaddingBytes: 0:U32
 // (op 594)   Stride(472): unitStride: false vgprBlockSize: 0
 // (op 594)   Stride(472): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 594)   Stride(472): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(64:U32, 4:U32)U32, 0:U32)U32
 // 	x = 256:U32

 // (op 1) ComputeIndex(594) END
// Allocated : 1 VGPR (Value: UInt32) (op 602): v4
v_and_b32 v4, 63, v1 // (op 602) 
 // (op 602) END: Flatten[1]
 // (op 602) BEGIN: Tile
 // (op 602) {Tile: BitwiseAnd(v3:U32, 1:U32)U32}
 // (op 602) BEGIN: Flatten[1]
 // (op 602) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 31:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 602): v5
v_and_b32 v5, 1, v3 // (op 602) 
 // (op 602) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 602): v3

// Allocated : 1 VGPR (Value: UInt32) (op 602): v3
v_and_b32 v3, 31, v1 // (op 602) 
 // (op 602) END: Flatten[1]
 // (op 602) LogicalShiftR({Flatten[1]: v4:U32}, 5:U32)U32
 // (op 602) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 602): v6
v_lshrrev_b32 v6, 5, v4 // (op 602) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 602): v4

// Allocated : 1 VGPR (Value: UInt32) (op 602): v4
v_lshl_add_u32 v4, v5, 5, v3 // (op 602) 
 // (op 602) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 602): v3

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 602): v5

 // (op 602) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 602): v3
v_lshl_add_u32 v3, v6, 6, v4 // (op 602) 
 // (op 602) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 602): v4

 // Freeing : 1 VGPR (Value: UInt32) (op 602): v6

 // (op 602) ShiftL({Tile: v3:U32}, 2:U32)U32
 // (op 602) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 602)         value (Tile: VGPR Value: UInt32 x 1: v3) 
 // (op 602)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated Offset602: 1 VGPR (Value: UInt32) (op 602): v4
v_lshlrev_b32 v4, 2, v3 // (op 602) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 602): v3

 // FastArithmetic:	orig = Add(Multiply(2:U32, 64:U32)U32, 0:U32)U32
 // 	x = 128:U32

 // (op 602)   Stride(478): indexExpr: 128:U32
 // (op 602)   Stride(478): indexExprPaddingBytes: 0:U32
 // (op 602)   Stride(478): unitStride: false vgprBlockSize: 0
 // (op 602)   Stride(478): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 602)   Stride(478): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(128:U32, 4:U32)U32, 0:U32)U32
 // 	x = 512:U32

 // (op 1) ComputeIndex(602) END
 // (op 1) ComputeIndex(603) BEGIN
 // (op 603) KernelGraph::LoadStoreTileGenerator::ComputeIndex(603): target 22 increment 159 base 477 offset 479 stride 480 buffer -1
 // FastArithmetic:	orig = Add(Multiply(1:U32, 64:U32)U32, 0:U32)U32
 // 	x = 64:U32

 // (op 603)   Stride(480): indexExpr: 64:U32
 // (op 603)   Stride(480): indexExprPaddingBytes: 0:U32
 // (op 603)   Stride(480): unitStride: false vgprBlockSize: 0
 // (op 603)   Stride(480): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 603)   Stride(480): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(64:U32, 4:U32)U32, 0:U32)U32
 // 	x = 256:U32

 // (op 1) ComputeIndex(603) END
 // (op 1) LoadVGPR Value: Float(2) BEGIN
 // tag 25: v**UNALLOCATED**
 // (op 2) GEN: LoadVGPR; scalar value
 // (op 2) Get arg user_Float_Value_8
 // (op 1) LoadVGPR Value: Float(24) BEGIN
 // tag 34: v**UNALLOCATED**
 // (op 24) GEN: LoadVGPR; scalar value
 // (op 24) Get arg user_Float_Value_6
// Allocated : 1 VGPR (Value: Float) (op 2): v3
s_waitcnt lgkmcnt(0)
v_mov_b32 v3, s17 // (op 2) Move value
 // (op 1) LoadVGPR Value: Float(2) END
// Allocated : 1 VGPR (Value: Float) (op 24): v5
v_mov_b32 v5, s16 // (op 24) Move value
 // (op 1) LoadVGPR Value: Float(24) END
 // (op 1) Deallocate{user_Float_Value_8}(786) BEGIN
 // (op 786) Deallocate user_Float_Value_8
 // Freeing user_Float_Value_8: 1 SGPR (Value: Raw32): s17

 // (op 1) Deallocate{user_Float_Value_8}(786) END
 // (op 1) Deallocate{user_Float_Value_6}(788) BEGIN
 // (op 788) Deallocate user_Float_Value_6
 // Freeing user_Float_Value_6: 1 SGPR (Value: Raw32): s16

 // (op 1) Deallocate{user_Float_Value_6}(788) END
 // (op 1) Scope(514) BEGIN
 // (op 514) Lock Scope 514
 // (op 514) generate({519})
 // (op 514) NOP(519) BEGIN
 // (op 519) generate({})
 // (op 519) end: generate({})
 // (op 514) NOP(519) END
 // (op 514) Assign ACCVGPR 0.00000:S(46) BEGIN
 // (op 46) Assign dim(36) = 0.00000:S
 // tag 36: a**UNALLOCATED**
 // (op 46) Generate 0.00000:S into DataFlowTag36: ACCVGPR Value: Float x 16: (unallocated)
// Allocated DataFlowTag36: 16 ACCVGPRs (Value: Float) (op 46): a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15
v_accvgpr_write a0, 0.00000 // (op 46) call()
v_accvgpr_write a1, 0.00000 // (op 46) call()
v_accvgpr_write a2, 0.00000 // (op 46) call()
v_accvgpr_write a3, 0.00000 // (op 46) call()
v_accvgpr_write a4, 0.00000 // (op 46) call()
v_accvgpr_write a5, 0.00000 // (op 46) call()
v_accvgpr_write a6, 0.00000 // (op 46) call()
v_accvgpr_write a7, 0.00000 // (op 46) call()
v_accvgpr_write a8, 0.00000 // (op 46) call()
v_accvgpr_write a9, 0.00000 // (op 46) call()
v_accvgpr_write a10, 0.00000 // (op 46) call()
v_accvgpr_write a11, 0.00000 // (op 46) call()
v_accvgpr_write a12, 0.00000 // (op 46) call()
v_accvgpr_write a13, 0.00000 // (op 46) call()
v_accvgpr_write a14, 0.00000 // (op 46) call()
v_accvgpr_write a15, 0.00000 // (op 46) call()
 // (op 514) Assign ACCVGPR 0.00000:S(46) END
 // (op 514) NOP(566) BEGIN
 // (op 566) generate({})
 // (op 566) end: generate({})
 // (op 514) NOP(566) END
 // (op 514) ComputeIndex(560) BEGIN
 // (op 560) KernelGraph::LoadStoreTileGenerator::ComputeIndex(560): target 1 increment 51 base -1 offset 443 stride 444 buffer 446
 // tag 443: v**UNALLOCATED**
 // FastArithmetic:	orig = {Split: Add(Multiply({Tile: Add(Multiply({Workgroup Index X: s2:U32}, 64:U32)U32, {Tile: Add(Multiply({Flatten[1]: Modulo({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32})U32}, 1:U64)U64, Multiply({Tile: Add(Multiply(0:U32, 64:U32)U32, {Tile: Add(Multiply({Flatten[0]: Divide({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32})U32}, Tensor_0_stride_1_10:I64)I64)U64}
 // 	x = {Split: Add(Convert({Tile: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({TileTile: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32}, Tensor_0_stride_1_10:I64)I64)U64}

 // (op 560)   Offset(443): indexExpr: {Split: Add(Convert({Tile: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({TileTile: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32}, Tensor_0_stride_1_10:I64)I64)U64}
 // (op 560)   Offset(443): paddingBytes: 0:U32
 // (op 560) Generate Convert(Add(Multiply({Split: Add(Convert({Tile: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({TileTile: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32}, Tensor_0_stride_1_10:I64)I64)U64}, 4:U32)U64, 0:U32)U64)U32 into Offset560: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Split: Add(Convert({Tile: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({TileTile: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32}, Tensor_0_stride_1_10:I64)I64)U64}, 4:U32)U64, 0:U32)U64)U32
 // 	x = Convert({Split: AddShiftL(Convert({Tile: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({TileTile: ShiftL(LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32, 2:U32)U32}, Tensor_0_stride_1_10:I64)I64, 2:U32)U64})U32

 // (op 560) Get arg Tensor_0_stride_1_10
 // (op 560) reg expression
 // (op 560) reg expression
 // (op 560) reg expression
 // (op 560) BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32
 // (op 560) LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 560): v6
v_and_b32 v6, -16, v1 // (op 560) 
// Allocated : 1 VGPR (Value: UInt32) (op 560): v7
v_lshrrev_b32 v7, 4, v1 // (op 560) 
 // (op 560) BEGIN: Flatten[1]
 // (op 560) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v6:U32)U32}
 // (op 560) BEGIN: TileTile
 // (op 560) {TileTile: ShiftL(v7:U32, 2:U32)U32}
 // (op 560) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 560)         value (VGPR Value: UInt32 x 1: v7) 
 // (op 560)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 560): v8
v_sub_u32 v8, v1, v6 // (op 560) 
 // (op 560) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 560): v6

// Allocated : 1 VGPR (Value: UInt32) (op 560): v6
v_lshlrev_b32 v6, 2, v7 // (op 560) 
 // (op 560) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 560): v7

 // (op 560) BEGIN: Tile
 // (op 560) {Tile: ShiftL({Flatten[1]: v8:U32}, 2:U32)U32}
 // (op 560) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 560)         value (Flatten[1]: VGPR Value: UInt32 x 1: v8) 
 // (op 560)         shiftAmount (Literal Value: UInt32 x 0: 2)
 // (op 560) Multiply({TileTile: v6:U32}, {Tensor_0_stride_1_10: s[22:23]:I64})I64
 // (op 560) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 560)           lhs (TileTile: VGPR Value: UInt32 x 1: v6) 
 // (op 560)           rhs (Tensor_0_stride_1_10: SGPR Value: Int64 x 1: s[22:23])
// Allocated : 1 VGPR (Value: UInt32) (op 560): v7
v_lshlrev_b32 v7, 2, v8 // (op 560) 
 // (op 560) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 560): v8

// Allocated : 2 VGPRs (Value: Int64) (op 560): v8, v9
v_mul_lo_u32 v9, v6, s23 // (op 560) most significant: low of low * high
 // (op 560) low of high * low omitted due to zero input.
 // (op 560) BEGIN: Tile
// Allocated : 1 VGPR (Value: Int32) (op 560): v10
v_mul_hi_u32 v10, v6, s22 // (op 560) most significant: high of low * low
v_mul_lo_u32 v8, v6, s22 // (op 560) least significant: low of low * low
v_add_u32 v9, v9, v10 // (op 560) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 560): v10

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 560): v6

// Allocated : 1 VGPR (Value: UInt32) (op 560): v6
v_lshl_add_u32 v6, s2, 6, v7 // (op 560) 
 // (op 560) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 560): v7

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 560): v10, v11
v_mov_b32 v11, 0 // (op 560) convert
v_mov_b32 v10, v6 // (op 560) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 560): v6

 // (op 560) BEGIN: Split
 // (op 560) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 560)      lhs (Tile: VGPR Value: UInt64 x 1: v[10:11]) 
 // (op 560)      rhs (VGPR Value: Int64 x 1: v[8:9])
// Allocated : 2 VGPRs (Value: UInt64) (op 560): v6, v7
// Allocated : 2 SGPRs (Value: Bool64) (op 560): s0, s1
v_add_co_u32 v6, s[0:1], v10, v8 // (op 560) least significant half
v_addc_co_u32 v7, s[0:1], v11, v9, s[0:1] // (op 560) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 560): s0, s1

 // (op 560) ShiftL: dest (VGPR Value: UInt64 x 1: v[6:7]) = 
 // (op 560)         value (VGPR Value: UInt64 x 1: v[6:7]) 
 // (op 560)         shiftAmount (Literal Value: UInt32 x 0: 2)
v_lshlrev_b64 v[6:7], 2, v[6:7] // (op 560) 
 // (op 560) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 560): v8, v9

 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 560): v10, v11

// Allocated Offset560: 1 VGPR (Value: UInt32) (op 560): v8
v_mov_b32 v8, v6 // (op 560) convert
 // Freeing Split: 2 VGPRs (Value: UInt64) (op 560): v6, v7

 // FastArithmetic:	orig = Add(Multiply(0:U32, 1:U64)U64, Multiply(64:U32, Tensor_0_stride_1_10:I64)I64)U64
 // 	x = Convert(ShiftL(Tensor_0_stride_1_10:I64, 6:U32)I64)U64

 // (op 560)   Stride(444): indexExpr: Convert(ShiftL(Tensor_0_stride_1_10:I64, 6:U32)I64)U64
 // (op 560)   Stride(444): indexExprPaddingBytes: 0:U32
 // (op 560)   Stride(444): unitStride: false vgprBlockSize: 0
 // (op 560)   Stride(444): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 560)   Stride(444): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(Convert(ShiftL(Tensor_0_stride_1_10:I64, 6:U32)I64)U64, 4:U32)U64, 0:U32)U64
 // 	x = ShiftL(Convert(ShiftL(Tensor_0_stride_1_10:I64, 6:U32)I64)U64, 2:U32)U64

 // tag 446: s**UNALLOCATED**
 // (op 560) Get arg Tensor_0_pointer
// Allocated Buffer446: 4 SGPRs (Buffer: None) (op 560): s36, s37, s38, s39
s_mov_b32 s36, s6 // (op 560) 
s_mov_b32 s37, s7 // (op 560) 
s_mov_b32 s39, 131072 // (op 560) default options
 // (op 560) Generate Multiply(Tensor_0_extent_0:I64, 4:U32)I64 into nullptr
 // FastArithmetic:	orig = Multiply(Tensor_0_extent_0:I64, 4:U32)I64
 // 	x = ShiftL(Tensor_0_extent_0:I64, 2:U32)I64

 // (op 560) Get arg Tensor_0_extent_0
 // (op 560) reg expression
 // (op 560) ShiftL({Tensor_0_extent_0: s[4:5]:I64}, 2:U32)I64
 // (op 560) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 560)         value (Tensor_0_extent_0: SGPR Value: Int64 x 1: s[4:5]) 
 // (op 560)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 560): s0, s1
s_lshl_b64 s[0:1], s[4:5], 2 // (op 560) 
s_mov_b32 s38, s0 // (op 560) 
 // Freeing : 2 SGPRs (Value: Int64) (op 560): s0, s1

 // (op 514)  Tag 560non referenced 	extraArgs = {Tensor_0_size_0_8, Tensor_0_size_1_9}

 // (op 514) ComputeIndex(560) END
 // (op 514) Deallocate{Tensor_0_pointer}(798) BEGIN
 // (op 798) Deallocate Tensor_0_pointer
 // Freeing Tensor_0_pointer: 2 SGPRs (Value: Raw32): s6, s7

 // (op 514) Deallocate{Tensor_0_pointer}(798) END
 // (op 514) ComputeIndex(562) BEGIN
 // (op 562) KernelGraph::LoadStoreTileGenerator::ComputeIndex(562): target 1 increment 61 base 443 offset 447 stride 448 buffer -1
 // FastArithmetic:	orig = Add(Multiply(0:U32, 1:U64)U64, Multiply(1:U32, Tensor_0_stride_1_10:I64)I64)U64
 // 	x = Convert(Tensor_0_stride_1_10:I64)U64

 // (op 562)   Stride(448): indexExpr: Convert(Tensor_0_stride_1_10:I64)U64
 // (op 562)   Stride(448): indexExprPaddingBytes: 0:U32
 // (op 562)   Stride(448): unitStride: false vgprBlockSize: 0
 // (op 562)   Stride(448): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 562)   Stride(448): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(Convert(Tensor_0_stride_1_10:I64)U64, 4:U32)U64, 0:U32)U64
 // 	x = ShiftL(Convert(Tensor_0_stride_1_10:I64)U64, 2:U32)U64

 // (op 514)  Tag 562non referenced 	extraArgs = {Tensor_0_extent_0, Tensor_0_size_0_8, Tensor_0_size_1_9, Tensor_0_stride_1_10}

 // (op 514) ComputeIndex(562) END
 // (op 514) ComputeIndex(563) BEGIN
 // (op 563) KernelGraph::LoadStoreTileGenerator::ComputeIndex(563): target 1 increment 62 base 447 offset 449 stride 450 buffer -1
 // FastArithmetic:	orig = Add(Multiply(1:U32, 1:U64)U64, Multiply(0:U32, Tensor_0_stride_1_10:I64)I64)U64
 // 	x = 1:U64

 // (op 563)   Stride(450): indexExpr: 1:U64
 // (op 563)   Stride(450): indexExprPaddingBytes: 0:U32
 // (op 563)   Stride(450): unitStride: true vgprBlockSize: 0
 // (op 563)   Stride(450): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 563)   Stride(450): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(1:U64, 4:U32)U64, 0:U32)U64
 // 	x = 4:U64

 // (op 514)  Tag 563non referenced 	extraArgs = {Tensor_0_extent_0, Tensor_0_size_0_8, Tensor_0_size_1_9, Tensor_0_stride_1_10}

 // (op 514) ComputeIndex(563) END
 // (op 514) ComputeIndex(571) BEGIN
 // (op 571) KernelGraph::LoadStoreTileGenerator::ComputeIndex(571): target 2 increment 110 base -1 offset 451 stride 452 buffer 454
 // tag 451: v**UNALLOCATED**
 // FastArithmetic:	orig = {Split: Add(Multiply({Tile: Add(Multiply(0:U32, 64:U32)U32, {Tile: Add(Multiply({Flatten[0]: Divide({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32})U32}, Tensor_2_stride_0_11:I64)I64, Multiply({Tile: Add(Multiply({Workgroup Index Y: s3:U32}, 64:U32)U32, {Tile: Add(Multiply({Flatten[1]: Modulo({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32})U32}, 1:U64)U64)U64}
 // 	x = {Split: MultiplyAdd({TileTile: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32}, Tensor_2_stride_0_11:I64, Convert({Tile: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64)U64}

 // (op 571)   Offset(451): indexExpr: {Split: MultiplyAdd({TileTile: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32}, Tensor_2_stride_0_11:I64, Convert({Tile: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64)U64}
 // (op 571)   Offset(451): paddingBytes: 0:U32
 // (op 571) Generate Convert(Add(Multiply({Split: MultiplyAdd({TileTile: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32}, Tensor_2_stride_0_11:I64, Convert({Tile: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64)U64}, 4:U32)U64, 0:U32)U64)U32 into Offset571: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Split: MultiplyAdd({TileTile: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32}, Tensor_2_stride_0_11:I64, Convert({Tile: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64)U64}, 4:U32)U64, 0:U32)U64)U32
 // 	x = Convert(ShiftL({Split: MultiplyAdd({TileTile: ShiftL(LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32, 2:U32)U32}, Tensor_2_stride_0_11:I64, Convert({Tile: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32)U32}, 2:U32)U32})U32})U64)U64}, 2:U32)U64)U32

 // (op 571) Get arg Tensor_2_stride_0_11
 // (op 571) reg expression
 // (op 571) reg expression
 // (op 571) reg expression
 // (op 571) BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32
 // (op 571) LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 571): v7
v_and_b32 v7, -16, v1 // (op 571) 
// Allocated : 1 VGPR (Value: UInt32) (op 571): v6
v_lshrrev_b32 v6, 4, v1 // (op 571) 
 // (op 571) BEGIN: Flatten[1]
 // (op 571) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v7:U32)U32}
 // (op 571) BEGIN: TileTile
 // (op 571) {TileTile: ShiftL(v6:U32, 2:U32)U32}
 // (op 571) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 571)         value (VGPR Value: UInt32 x 1: v6) 
 // (op 571)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 571): v9
v_sub_u32 v9, v1, v7 // (op 571) 
 // (op 571) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 571): v7

// Allocated : 1 VGPR (Value: UInt32) (op 571): v7
v_lshlrev_b32 v7, 2, v6 // (op 571) 
 // (op 571) END: TileTile
 // Freeing : 1 VGPR (Value: UInt32) (op 571): v6

 // (op 571) BEGIN: Tile
 // (op 571) {Tile: ShiftL({Flatten[1]: v9:U32}, 2:U32)U32}
 // (op 571) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 571)         value (Flatten[1]: VGPR Value: UInt32 x 1: v9) 
 // (op 571)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 571): v6
v_lshlrev_b32 v6, 2, v9 // (op 571) 
 // (op 571) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 571): v9

 // (op 571) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 571): v9
v_lshl_add_u32 v9, s3, 6, v6 // (op 571) 
 // (op 571) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 571): v6

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 571): v10, v11
v_mov_b32 v11, 0 // (op 571) convert
v_mov_b32 v10, v9 // (op 571) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 571): v9

 // (op 571) BEGIN: Split
 // (op 571) Multiply: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 571)           lhs (TileTile: VGPR Value: UInt32 x 1: v7) 
 // (op 571)           rhs (Tensor_2_stride_0_11: SGPR Value: Int64 x 1: s[24:25])
// Allocated : 2 VGPRs (Value: UInt64) (op 571): v12, v13
v_mul_lo_u32 v13, v7, s25 // (op 571) most significant: low of low * high
 // (op 571) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 571): v6
v_mul_hi_u32 v6, v7, s24 // (op 571) most significant: high of low * low
v_mul_lo_u32 v12, v7, s24 // (op 571) least significant: low of low * low
v_add_u32 v13, v13, v6 // (op 571) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 571): v6

 // (op 571) Add: dest (VGPR Value: UInt64 x 1: v[12:13]) = 
 // (op 571)      lhs (VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 571)      rhs (Tile: VGPR Value: UInt64 x 1: v[10:11])
// Allocated : 2 SGPRs (Value: Bool64) (op 571): s0, s1
v_add_co_u32 v12, s[0:1], v12, v10 // (op 571) least significant half
v_addc_co_u32 v13, s[0:1], v13, v11, s[0:1] // (op 571) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 571): s0, s1

 // (op 571) END: Split
 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 571): v10, v11

 // Freeing TileTile: 1 VGPR (Value: UInt32) (op 571): v7

 // (op 571) ShiftL({Split: v[12:13]:U64}, 2:U32)U64
 // (op 571) ShiftL: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 571)         value (Split: VGPR Value: UInt64 x 1: v[12:13]) 
 // (op 571)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 VGPRs (Value: UInt64) (op 571): v6, v7
v_lshlrev_b64 v[6:7], 2, v[12:13] // (op 571) 
 // Freeing Split: 2 VGPRs (Value: UInt64) (op 571): v12, v13

// Allocated Offset571: 1 VGPR (Value: UInt32) (op 571): v9
v_mov_b32 v9, v6 // (op 571) convert
 // Freeing : 2 VGPRs (Value: UInt64) (op 571): v6, v7

 // FastArithmetic:	orig = Add(Multiply(64:U32, Tensor_2_stride_0_11:I64)I64, Multiply(0:U32, 1:U64)U64)U64
 // 	x = Convert(ShiftL(Tensor_2_stride_0_11:I64, 6:U32)I64)U64

 // (op 571)   Stride(452): indexExpr: Convert(ShiftL(Tensor_2_stride_0_11:I64, 6:U32)I64)U64
 // (op 571)   Stride(452): indexExprPaddingBytes: 0:U32
 // (op 571)   Stride(452): unitStride: false vgprBlockSize: 0
 // (op 571)   Stride(452): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 571)   Stride(452): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(Convert(ShiftL(Tensor_2_stride_0_11:I64, 6:U32)I64)U64, 4:U32)U64, 0:U32)U64
 // 	x = ShiftL(Convert(ShiftL(Tensor_2_stride_0_11:I64, 6:U32)I64)U64, 2:U32)U64

 // tag 454: s**UNALLOCATED**
 // (op 571) Get arg Tensor_2_pointer
// Allocated Buffer454: 4 SGPRs (Buffer: None) (op 571): s40, s41, s42, s43
s_mov_b32 s40, s10 // (op 571) 
s_mov_b32 s41, s11 // (op 571) 
s_mov_b32 s43, 131072 // (op 571) default options
 // (op 571) Generate Multiply(Tensor_2_extent_2:I64, 4:U32)I64 into nullptr
 // FastArithmetic:	orig = Multiply(Tensor_2_extent_2:I64, 4:U32)I64
 // 	x = ShiftL(Tensor_2_extent_2:I64, 2:U32)I64

 // (op 571) Get arg Tensor_2_extent_2
 // (op 571) reg expression
 // (op 571) ShiftL({Tensor_2_extent_2: s[8:9]:I64}, 2:U32)I64
 // (op 571) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 571)         value (Tensor_2_extent_2: SGPR Value: Int64 x 1: s[8:9]) 
 // (op 571)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 571): s0, s1
s_lshl_b64 s[0:1], s[8:9], 2 // (op 571) 
s_mov_b32 s42, s0 // (op 571) 
 // Freeing : 2 SGPRs (Value: Int64) (op 571): s0, s1

 // (op 514)  Tag 571non referenced 	extraArgs = {Tensor_0_size_1_9, Tensor_2_size_1_12}

 // (op 514) ComputeIndex(571) END
 // (op 514) Deallocate{Tensor_2_pointer}(800) BEGIN
 // (op 800) Deallocate Tensor_2_pointer
 // Freeing Tensor_2_pointer: 2 SGPRs (Value: Raw32): s10, s11

 // (op 514) Deallocate{Tensor_2_pointer}(800) END
 // (op 514) ComputeIndex(573) BEGIN
 // (op 573) KernelGraph::LoadStoreTileGenerator::ComputeIndex(573): target 2 increment 121 base 451 offset 455 stride 456 buffer -1
 // FastArithmetic:	orig = Add(Multiply(1:U32, Tensor_2_stride_0_11:I64)I64, Multiply(0:U32, 1:U64)U64)U64
 // 	x = Convert(Tensor_2_stride_0_11:I64)U64

 // (op 573)   Stride(456): indexExpr: Convert(Tensor_2_stride_0_11:I64)U64
 // (op 573)   Stride(456): indexExprPaddingBytes: 0:U32
 // (op 573)   Stride(456): unitStride: false vgprBlockSize: 0
 // (op 573)   Stride(456): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 573)   Stride(456): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(Convert(Tensor_2_stride_0_11:I64)U64, 4:U32)U64, 0:U32)U64
 // 	x = ShiftL(Convert(Tensor_2_stride_0_11:I64)U64, 2:U32)U64

 // (op 514)  Tag 573non referenced 	extraArgs = {Tensor_0_size_1_9, Tensor_2_extent_2, Tensor_2_size_1_12, Tensor_2_stride_0_11}

 // (op 514) ComputeIndex(573) END
 // (op 514) ComputeIndex(574) BEGIN
 // (op 574) KernelGraph::LoadStoreTileGenerator::ComputeIndex(574): target 2 increment 122 base 455 offset 457 stride 458 buffer -1
 // FastArithmetic:	orig = Add(Multiply(0:U32, Tensor_2_stride_0_11:I64)I64, Multiply(1:U32, 1:U64)U64)U64
 // 	x = 1:U64

 // (op 574)   Stride(458): indexExpr: 1:U64
 // (op 574)   Stride(458): indexExprPaddingBytes: 0:U32
 // (op 574)   Stride(458): unitStride: true vgprBlockSize: 0
 // (op 574)   Stride(458): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 574)   Stride(458): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(1:U64, 4:U32)U64, 0:U32)U64
 // 	x = 4:U64

 // (op 514)  Tag 574non referenced 	extraArgs = {Tensor_0_size_1_9, Tensor_2_extent_2, Tensor_2_size_1_12, Tensor_2_stride_0_11}

 // (op 514) ComputeIndex(574) END
 // (op 514) ComputeIndex(588) BEGIN
 // (op 588) KernelGraph::LoadStoreTileGenerator::ComputeIndex(588): target 14 increment 76 base -1 offset 465 stride 466 buffer -1
 // tag 465: v**UNALLOCATED**
 // FastArithmetic:	orig = {Flatten: Add(Multiply({Flatten: Add(Multiply({Tile[0]: Divide({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32}, 64:U32)U32, {Flatten: Add(Multiply({Tile[1]: Modulo({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32})U32}
 // 	x = {Flatten: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}

 // (op 588)   Offset(465): indexExpr: {Flatten: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}
 // (op 588)   Offset(465): paddingBytes: 0:U32
 // (op 588) Generate Convert(Add(Multiply({Flatten: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}, 4:U32)U32, 0:U32)U32)U32 into Offset588: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Flatten: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}, 4:U32)U32, 0:U32)U32)U32
 // 	x = ShiftL({Flatten: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32, 8:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32)U32}, 2:U32)U32})U32}, 2:U32)U32

 // (op 588) reg expression
 // (op 588) BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32
 // (op 588) LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 588): v7
v_and_b32 v7, -16, v1 // (op 588) 
// Allocated : 1 VGPR (Value: UInt32) (op 588): v6
v_lshrrev_b32 v6, 4, v1 // (op 588) 
 // (op 588) BEGIN: Tile[1]
 // (op 588) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v7:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 588): v10
v_sub_u32 v10, v1, v7 // (op 588) 
 // (op 588) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 588): v7

 // (op 588) BEGIN: Flatten
 // (op 588) {Flatten: ShiftL({Tile[1]: v10:U32}, 2:U32)U32}
 // (op 588) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 588)         value (Tile[1]: VGPR Value: UInt32 x 1: v10) 
 // (op 588)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 588): v7
v_lshlrev_b32 v7, 2, v10 // (op 588) 
 // (op 588) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 588): v10

 // (op 588) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 588): v10
v_lshl_add_u32 v10, v6, 8, v7 // (op 588) 
 // (op 588) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 588): v7

 // Freeing : 1 VGPR (Value: UInt32) (op 588): v6

 // (op 588) ShiftL({Flatten: v10:U32}, 2:U32)U32
 // (op 588) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 588)         value (Flatten: VGPR Value: UInt32 x 1: v10) 
 // (op 588)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated Offset588: 1 VGPR (Value: UInt32) (op 588): v7
v_lshlrev_b32 v7, 2, v10 // (op 588) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 588): v10

 // FastArithmetic:	orig = Add(Multiply(1:I, Multiply(1:I, 64:U32)U32)U32, Multiply(0:I, 1:I)I)U32
 // 	x = 64:U32

 // (op 588)   Stride(466): indexExpr: 64:U32
 // (op 588)   Stride(466): indexExprPaddingBytes: 0:U32
 // (op 588)   Stride(466): unitStride: false vgprBlockSize: 0
 // (op 588)   Stride(466): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 588)   Stride(466): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(64:U32, 4:U32)U32, 0:U32)U32
 // 	x = 256:U32

 // (op 514) ComputeIndex(588) END
 // (op 514) ComputeIndex(589) BEGIN
 // (op 589) KernelGraph::LoadStoreTileGenerator::ComputeIndex(589): target 14 increment 77 base 465 offset 467 stride 468 buffer -1
 // FastArithmetic:	orig = Add(Multiply(0:I, Multiply(1:I, 64:U32)U32)U32, Multiply(1:I, 1:I)I)U32
 // 	x = 1:U32

 // (op 589)   Stride(468): indexExpr: 1:U32
 // (op 589)   Stride(468): indexExprPaddingBytes: 0:U32
 // (op 589)   Stride(468): unitStride: true vgprBlockSize: 0
 // (op 589)   Stride(468): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 589)   Stride(468): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(1:U32, 4:U32)U32, 0:U32)U32
 // 	x = 4:U32

 // (op 514) ComputeIndex(589) END
 // (op 514) ComputeIndex(597) BEGIN
 // (op 597) KernelGraph::LoadStoreTileGenerator::ComputeIndex(597): target 22 increment 136 base -1 offset 473 stride 474 buffer -1
 // tag 473: v**UNALLOCATED**
 // FastArithmetic:	orig = {Flatten: Add(Multiply({Flatten: Add(Multiply({Tile[0]: Divide({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32}, 64:U32)U32, {Flatten: Add(Multiply({Tile[1]: Modulo({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32})U32}
 // 	x = {Flatten: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}

 // (op 597)   Offset(473): indexExpr: {Flatten: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}
 // (op 597)   Offset(473): paddingBytes: 0:U32
 // (op 597) Generate Convert(Add(Multiply({Flatten: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}, 4:U32)U32, 0:U32)U32)U32 into Offset597: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Flatten: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}, 4:U32)U32, 0:U32)U32)U32
 // 	x = ShiftL({Flatten: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32, 8:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32)U32}, 2:U32)U32})U32}, 2:U32)U32

 // (op 597) reg expression
 // (op 597) BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32
 // (op 597) LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 597): v6
v_and_b32 v6, -16, v1 // (op 597) 
// Allocated : 1 VGPR (Value: UInt32) (op 597): v10
v_lshrrev_b32 v10, 4, v1 // (op 597) 
 // (op 597) BEGIN: Tile[1]
 // (op 597) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v6:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 597): v11
v_sub_u32 v11, v1, v6 // (op 597) 
 // (op 597) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 597): v6

 // (op 597) BEGIN: Flatten
 // (op 597) {Flatten: ShiftL({Tile[1]: v11:U32}, 2:U32)U32}
 // (op 597) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 597)         value (Tile[1]: VGPR Value: UInt32 x 1: v11) 
 // (op 597)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 597): v6
v_lshlrev_b32 v6, 2, v11 // (op 597) 
 // (op 597) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 597): v11

 // (op 597) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 597): v11
v_lshl_add_u32 v11, v10, 8, v6 // (op 597) 
 // (op 597) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 597): v6

 // Freeing : 1 VGPR (Value: UInt32) (op 597): v10

 // (op 597) ShiftL({Flatten: v11:U32}, 2:U32)U32
 // (op 597) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 597)         value (Flatten: VGPR Value: UInt32 x 1: v11) 
 // (op 597)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated Offset597: 1 VGPR (Value: UInt32) (op 597): v6
v_lshlrev_b32 v6, 2, v11 // (op 597) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 597): v11

 // FastArithmetic:	orig = Add(Multiply(1:I, Multiply(1:I, 64:U32)U32)U32, Multiply(0:I, 1:I)I)U32
 // 	x = 64:U32

 // (op 597)   Stride(474): indexExpr: 64:U32
 // (op 597)   Stride(474): indexExprPaddingBytes: 0:U32
 // (op 597)   Stride(474): unitStride: false vgprBlockSize: 0
 // (op 597)   Stride(474): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 597)   Stride(474): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(64:U32, 4:U32)U32, 0:U32)U32
 // 	x = 256:U32

 // (op 514) ComputeIndex(597) END
 // (op 514) ComputeIndex(598) BEGIN
 // (op 598) KernelGraph::LoadStoreTileGenerator::ComputeIndex(598): target 22 increment 137 base 473 offset 475 stride 476 buffer -1
 // FastArithmetic:	orig = Add(Multiply(0:I, Multiply(1:I, 64:U32)U32)U32, Multiply(1:I, 1:I)I)U32
 // 	x = 1:U32

 // (op 598)   Stride(476): indexExpr: 1:U32
 // (op 598)   Stride(476): indexExprPaddingBytes: 0:U32
 // (op 598)   Stride(476): unitStride: true vgprBlockSize: 0
 // (op 598)   Stride(476): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 598)   Stride(476): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(1:U32, 4:U32)U32, 0:U32)U32
 // 	x = 4:U32

 // (op 514) ComputeIndex(598) END
 // (op 514) ForLoopOp KLoop: LessThan(DataFlowTag(277)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_9:I64, 6:U32)I64)U32)BL(41) BEGIN
 // (op 41) Initialize For Loop
 // (op 41) generate({42})
 // (op 41) Assign SGPR 0:U32(42) BEGIN
 // (op 42) Assign dim(277) = 0:U32
 // tag 277: s**UNALLOCATED**
 // (op 42) Generate 0:U32 into DataFlowTag277: SGPR Value: UInt32 x 1: (unallocated)
// Allocated DataFlowTag277: 1 SGPR (Value: UInt32) (op 42): s1
s_mov_b32 s1, 0 // (op 42) call()
 // (op 41) Assign SGPR 0:U32(42) END
 // (op 41) end: generate({42})
 // (op 41) Lock For Loop
 // (op 41) Generate LessThan(DataFlowTag(277)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_9:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 41) Get arg Tensor_0_size_1_9
 // (op 41) LessThan({DataFlowTag277: s1:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_9: s[20:21]:I64}, 6:U32)I64)U32)BL
 // (op 41) ArithmeticShiftR({Tensor_0_size_1_9: s[20:21]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 41): s6, s7
s_ashr_i64 s[6:7], s[20:21], 6 // (op 41) 
// Allocated : 1 SGPR (Value: UInt32) (op 41): s0
s_mov_b32 s0, s6 // (op 41) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 41): s6, s7

s_cmp_lt_u32 s1, s0 // (op 41) 
 // Freeing : 1 SGPR (Value: UInt32) (op 41): s0

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 41) 
s_cbranch_scc0 GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ForLoopBottom_KLoop_41 // (op 41) Condition: Top (jump to GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ForLoopBottom_KLoop_41 if false)
GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ForLoopTop_KLoop_41:
 // (op 41) 
 // (op 41) generate({4, 43})
 // (op 41) LoadTiled Value: Float(4) BEGIN
 // (op 4) GEN: loadMacroTileVGPRCI Value: Float
 // (op 4) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 4) Generate ShiftL(Convert(Tensor_0_stride_1_10:I64)U64, 2:U32)U64 into VGPR Value: UInt64 x 1: (unallocated)
 // (op 4) Get arg Tensor_0_stride_1_10
 // (op 4) reg expression
 // (op 4) convert
 // (op 4) ShiftL({ convertInPlaceTensor_0_stride_1_10: s[22:23]:U64}, 2:U32)U64
 // (op 4) ShiftL: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 4)         value ( convertInPlaceTensor_0_stride_1_10: SGPR Value: UInt64 x 1: s[22:23]) 
 // (op 4)         shiftAmount (Literal Value: UInt32 x 0: 2)
 // (op 41) Assign SGPR Add(DataFlowTag(277)U32, 1:U32)U32(43) BEGIN
 // (op 43) Assign dim(277) = Add(DataFlowTag(277)U32, 1:U32)U32
 // (op 43) Generate Add(DataFlowTag(277)U32, 1:U32)U32 into DataFlowTag277: SGPR Value: UInt32 x 1: s1
 // (op 43) reg expression
 // (op 43) Add({DataFlowTag277: s1:U32}, 1:U32)U32
s_add_u32 s1, s1, 1 // (op 43) 
 // (op 41) Assign SGPR Add(DataFlowTag(277)U32, 1:U32)U32(43) END
// Allocated : 2 VGPRs (Value: UInt64) (op 4): v10, v11
v_lshlrev_b64 v[10:11], 2, s[22:23] // (op 4) 
 // (op 4) Generate 4:U64 into nullptr
 // tag 6: v**UNALLOCATED**
 // (op 4) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = VGPR Value: UInt64 x 1: v[10:11]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = Buffer446: SGPR Buffer: None x 1: s[36:39]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 447: v**UNALLOCATED**
// Allocated offset447: 1 VGPR (Value: UInt32) (op 4): v12
v_mov_b32 v12, v8 // (op 4) 
// Allocated : 1 VGPR (Value: UInt32) (op 4): v13
v_mov_b32 v13, v12 // (op 4) 
 // (op 4)   M 4 N 4 elementsPerMove 4 bytesPerMove 16 rowStride v[10:11]:U64 colStride 4:U64 vgprBlockSize 0 numVGPRBlocks 1
// Allocated : 16 VGPRs (Value: Float) (op 4): v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29
buffer_load_dwordx4 v[14:17], v13, s[36:39], 0 offen // (op 4) Load value
// VMEM: Expected complete at 193 (current 118)
 // (op 4) Generate Add(v13:U32, v10:R)U32 into VGPR Value: UInt32 x 1: v13
 // (op 4) reg expression
 // (op 4) reg expression
 // (op 4) Add(v13:U32, v10:R)U32
v_add_u32 v13, v13, v10 // (op 4) 
buffer_load_dwordx4 v[18:21], v13, s[36:39], 0 offen // (op 4) Load value
// VMEM: Expected complete at 195 (current 120)
 // (op 4) Generate Add(v13:U32, v10:R)U32 into VGPR Value: UInt32 x 1: v13
 // (op 4) reg expression
 // (op 4) reg expression
 // (op 4) Add(v13:U32, v10:R)U32
v_add_u32 v13, v13, v10 // (op 4) 
buffer_load_dwordx4 v[22:25], v13, s[36:39], 0 offen // (op 4) Load value
// VMEM: Expected complete at 197 (current 122)
 // (op 4) Generate Add(v13:U32, v10:R)U32 into VGPR Value: UInt32 x 1: v13
 // (op 4) reg expression
 // (op 4) reg expression
 // (op 4) Add(v13:U32, v10:R)U32
v_add_u32 v13, v13, v10 // (op 4) 
buffer_load_dwordx4 v[26:29], v13, s[36:39], 0 offen // (op 4) Load value
// VMEM: Expected complete at 199 (current 124)
 // Freeing : 2 VGPRs (Value: UInt64) (op 4): v10, v11

 // Freeing : 1 VGPR (Value: UInt32) (op 4): v13

 // (op 41) LoadTiled Value: Float(4) END
 // (op 41) LoadTiled Value: Float(10) BEGIN
 // (op 10) GEN: loadMacroTileVGPRCI Value: Float
 // (op 10) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 10) Generate ShiftL(Convert(Tensor_2_stride_0_11:I64)U64, 2:U32)U64 into VGPR Value: UInt64 x 1: (unallocated)
 // (op 10) Get arg Tensor_2_stride_0_11
 // (op 10) reg expression
 // (op 10) convert
 // (op 10) ShiftL({ convertInPlaceTensor_2_stride_0_11: s[24:25]:U64}, 2:U32)U64
 // (op 10) ShiftL: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 10)         value ( convertInPlaceTensor_2_stride_0_11: SGPR Value: UInt64 x 1: s[24:25]) 
 // (op 10)         shiftAmount (Literal Value: UInt32 x 0: 2)
 // (op 41) Barrier(534) BEGIN
 // (op 41) Assign VGPR Convert(Add(DataFlowTag(443)U32, DataFlowTag(444)U64)U64)U32(561) BEGIN
 // (op 561) Assign dim(443) = Convert(Add(DataFlowTag(443)U32, DataFlowTag(444)U64)U64)U32
 // (op 561) Generate Convert(Add(DataFlowTag(443)U32, DataFlowTag(444)U64)U64)U32 into Offset560Split: VGPR Value: UInt32 x 1: v8
 // (op 561) Get arg Tensor_0_stride_1_10
 // (op 561) reg expression
 // (op 561) reg expression
 // (op 561) ShiftL({Tensor_0_stride_1_10: s[22:23]:I64}, 6:U32)I64
 // (op 561) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 561)         value (Tensor_0_stride_1_10: SGPR Value: Int64 x 1: s[22:23]) 
 // (op 561)         shiftAmount (Literal Value: UInt32 x 0: 6)
s_barrier  // (op 534) 
 // (op 41) Barrier(534) END
// Allocated : 2 VGPRs (Value: UInt64) (op 10): v10, v11
v_lshlrev_b64 v[10:11], 2, s[24:25] // (op 10) 
 // (op 10) Generate 4:U64 into nullptr
 // tag 7: v**UNALLOCATED**
 // (op 10) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = VGPR Value: UInt64 x 1: v[10:11]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = Buffer454: SGPR Buffer: None x 1: s[40:43]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

 // tag 455: v**UNALLOCATED**
 // (op 41) StoreLDSTile Value: Float(6) BEGIN
 // (op 6) GEN: StoreLDSTile
 // (op 6) GEN: storeMacroTileLDS OP 6 LDS 14 MacroTile 6
 // (op 6) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 6) Generate 256:U32 into nullptr
 // (op 6) Generate 4:U32 into nullptr
 // (op 6) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: v[14:29]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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

 // (op 6) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: v[14:29]
 // 	info.rowOffsetReg = Offset588: VGPR Value: UInt32 x 1: v7
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // (op 6) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 256
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated offset455: 1 VGPR (Value: UInt32) (op 10): v13
v_mov_b32 v13, v9 // (op 10) 
// Allocated : 1 VGPR (Value: UInt32) (op 10): v30
v_mov_b32 v30, v13 // (op 10) 
 // (op 10)   M 4 N 4 elementsPerMove 4 bytesPerMove 16 rowStride v[10:11]:U64 colStride 4:U64 vgprBlockSize 0 numVGPRBlocks 1
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 561): s6, s7
s_lshl_b64 s[6:7], s[22:23], 6 // (op 561) 
 // (op 561) convert
 // (op 561) ShiftL({ convertInPlace: s[6:7]:U64}, 2:U32)U64
 // (op 561) ShiftL: dest (SGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 561)         value ( convertInPlace: SGPR Value: UInt64 x 1: s[6:7]) 
 // (op 561)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: UInt64) (op 561): s10, s11
s_lshl_b64 s[10:11], s[6:7], 2 // (op 561) 
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 561): s6, s7

 // (op 561) Add({Offset560Split: v8:U32}, s[10:11]:U64)U64
 // (op 561) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 561)      lhs (Offset560Split: VGPR Value: UInt32 x 1: v8) 
 // (op 561)      rhs (SGPR Value: UInt64 x 1: s[10:11])
// Allocated : 1 VGPR (Value: Raw32) (op 561): v31
v_mov_b32 v31, s10 // (op 561) 
// Allocated : 1 VGPR (Value: Raw32) (op 561): v32
v_mov_b32 v32, s11 // (op 561) 
// Allocated : 2 VGPRs (Value: UInt64) (op 561): v34, v35
// Allocated : 2 SGPRs (Value: Bool64) (op 561): s6, s7
v_add_co_u32 v34, s[6:7], v8, v31 // (op 561) least significant half
v_addc_co_u32 v35, s[6:7], 0, v32, s[6:7] // (op 561) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 561): s6, s7

 // Freeing : 1 VGPR (Value: Raw32) (op 561): v32

 // Freeing : 1 VGPR (Value: Raw32) (op 561): v31

 // Freeing : 2 SGPRs (Value: UInt64) (op 561): s10, s11

v_mov_b32 v8, v34 // (op 561) convert
// Read-only register cannot be the destination of an instruction: VGPR Value: Raw32 x 1: v8
 // Freeing : 2 VGPRs (Value: UInt64) (op 561): v34, v35

 // (op 41)  Tag 561non referenced 	extraArgs = {Tensor_0_extent_0, Tensor_0_size_0_8, Tensor_0_size_1_9}

 // (op 41) Assign VGPR Convert(Add(DataFlowTag(443)U32, DataFlowTag(444)U64)U64)U32(561) END
s_waitcnt vmcnt(3)
ds_write_b128 v7, v[14:17] // (op 6) Store local data 
// VMEM: Expected complete at 157 (current 137)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
// Allocated : 16 VGPRs (Value: Float) (op 10): v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47
buffer_load_dwordx4 v[32:35], v30, s[40:43], 0 offen // (op 10) Load value
// VMEM: Expected complete at 271 (current 196)
 // (op 10) Generate Add(v30:U32, v10:R)U32 into VGPR Value: UInt32 x 1: v30
 // (op 10) reg expression
 // (op 10) reg expression
 // (op 10) Add(v30:U32, v10:R)U32
v_add_u32 v30, v30, v10 // (op 10) 
buffer_load_dwordx4 v[36:39], v30, s[40:43], 0 offen // (op 10) Load value
// VMEM: Expected complete at 273 (current 198)
 // (op 10) Generate Add(v30:U32, v10:R)U32 into VGPR Value: UInt32 x 1: v30
 // (op 10) reg expression
 // (op 10) reg expression
 // (op 10) Add(v30:U32, v10:R)U32
v_add_u32 v30, v30, v10 // (op 10) 
buffer_load_dwordx4 v[40:43], v30, s[40:43], 0 offen // (op 10) Load value
// VMEM: Expected complete at 275 (current 200)
 // (op 10) Generate Add(v30:U32, v10:R)U32 into VGPR Value: UInt32 x 1: v30
 // (op 10) reg expression
 // (op 10) reg expression
 // (op 10) Add(v30:U32, v10:R)U32
v_add_u32 v30, v30, v10 // (op 10) 
buffer_load_dwordx4 v[44:47], v30, s[40:43], 0 offen // (op 10) Load value
// VMEM: Expected complete at 277 (current 202)
 // Freeing : 2 VGPRs (Value: UInt64) (op 10): v10, v11

 // Freeing : 1 VGPR (Value: UInt32) (op 10): v30

 // (op 41) LoadTiled Value: Float(10) END
s_waitcnt vmcnt(6)
ds_write_b128 v7, v[18:21] offset:256 // (op 6) Store local data 
// VMEM: Expected complete at 166 (current 146)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 41) Assign VGPR Convert(Add(DataFlowTag(451)U32, DataFlowTag(452)U64)U64)U32(572) BEGIN
 // (op 572) Assign dim(451) = Convert(Add(DataFlowTag(451)U32, DataFlowTag(452)U64)U64)U32
 // (op 572) Generate Convert(Add(DataFlowTag(451)U32, DataFlowTag(452)U64)U64)U32 into Offset571: VGPR Value: UInt32 x 1: v9
 // (op 572) Get arg Tensor_2_stride_0_11
 // (op 572) reg expression
 // (op 572) reg expression
 // (op 572) ShiftL({Tensor_2_stride_0_11: s[24:25]:I64}, 6:U32)I64
 // (op 572) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 572)         value (Tensor_2_stride_0_11: SGPR Value: Int64 x 1: s[24:25]) 
 // (op 572)         shiftAmount (Literal Value: UInt32 x 0: 6)
// Allocated  convertInPlace: 2 SGPRs (Value: Int64) (op 572): s6, s7
s_lshl_b64 s[6:7], s[24:25], 6 // (op 572) 
 // (op 572) convert
 // (op 572) ShiftL({ convertInPlace: s[6:7]:U64}, 2:U32)U64
 // (op 572) ShiftL: dest (SGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 572)         value ( convertInPlace: SGPR Value: UInt64 x 1: s[6:7]) 
 // (op 572)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: UInt64) (op 572): s10, s11
s_lshl_b64 s[10:11], s[6:7], 2 // (op 572) 
 // Freeing  convertInPlace: 2 SGPRs (Value: Int64) (op 572): s6, s7

 // (op 572) Add({Offset571: v9:U32}, s[10:11]:U64)U64
 // (op 572) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 572)      lhs (Offset571: VGPR Value: UInt32 x 1: v9) 
 // (op 572)      rhs (SGPR Value: UInt64 x 1: s[10:11])
// Allocated : 1 VGPR (Value: Raw32) (op 572): v11
v_mov_b32 v11, s10 // (op 572) 
// Allocated : 1 VGPR (Value: Raw32) (op 572): v10
v_mov_b32 v10, s11 // (op 572) 
// Allocated : 2 VGPRs (Value: UInt64) (op 572): v30, v31
// Allocated : 2 SGPRs (Value: Bool64) (op 572): s6, s7
v_add_co_u32 v30, s[6:7], v9, v11 // (op 572) least significant half
v_addc_co_u32 v31, s[6:7], 0, v10, s[6:7] // (op 572) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 572): s6, s7

 // Freeing : 1 VGPR (Value: Raw32) (op 572): v10

 // Freeing : 1 VGPR (Value: Raw32) (op 572): v11

 // Freeing : 2 SGPRs (Value: UInt64) (op 572): s10, s11

v_mov_b32 v9, v30 // (op 572) convert
// Read-only register cannot be the destination of an instruction: VGPR Value: Raw32 x 1: v9
 // Freeing : 2 VGPRs (Value: UInt64) (op 572): v30, v31

 // (op 41)  Tag 572non referenced 	extraArgs = {Tensor_0_size_1_9, Tensor_2_extent_2, Tensor_2_size_1_12}

 // (op 41) Assign VGPR Convert(Add(DataFlowTag(451)U32, DataFlowTag(452)U64)U64)U32(572) END
s_waitcnt vmcnt(5)
ds_write_b128 v7, v[22:25] offset:512 // (op 6) Store local data 
// VMEM: Expected complete at 175 (current 155)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
s_waitcnt vmcnt(4)
ds_write_b128 v7, v[26:29] offset:768 // (op 6) Store local data 
// VMEM: Expected complete at 177 (current 157)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 41) StoreLDSTile Value: Float(6) END
 // (op 41) Deallocate{}(652) BEGIN
 // (op 652) Deallocate 6
 // Freeing : 16 VGPRs (Value: Float) (op 4): v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29

 // Deleting tag 6
 // (op 41) Deallocate{}(652) END
 // (op 41) Barrier(535) BEGIN
s_waitcnt lgkmcnt(0)
s_barrier  // (op 535) 
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 41) Barrier(535) END
 // (op 41) SetCoordinate(49) BEGIN
 // (op 49) SetCoordinate (49): Coordinate 282 = 0:U32
 // (op 49) generate({})
 // (op 49) end: generate({})
 // (op 49) generate({51})
 // (op 49) LoadLDSTile{Value: Float}(51) BEGIN
 // (op 51) GEN: loadMacroTileWAVELDS OP 51 LDS 14 WaveTile 87
 // (op 51) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 51) Generate 256:U32 into nullptr
 // tag 12: v**UNALLOCATED**
 // (op 51) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 51) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // (op 41) Barrier(541) BEGIN
s_barrier  // (op 541) 
 // (op 41) Barrier(541) END
// Allocated : 1 VGPR (Value: Float) (op 51): v11
ds_read_b32 v11, v0 // (op 51) Load local data 
// VMEM: Expected complete at 201 (current 181)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 49) LoadLDSTile{Value: Float}(51) END
 // (op 49) end: generate({51})
 // (op 41) SetCoordinate(49) END
 // (op 41) StoreLDSTile Value: Float(13) BEGIN
 // (op 13) GEN: StoreLDSTile
 // (op 13) GEN: storeMacroTileLDS OP 13 LDS 22 MacroTile 7
 // (op 13) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 13) Generate 256:U32 into nullptr
 // (op 13) Generate 4:U32 into nullptr
 // (op 13) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: v[32:47]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
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
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: v[32:47]
 // 	info.rowOffsetReg = Offset597: VGPR Value: UInt32 x 1: v6
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 13) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 256
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

s_waitcnt vmcnt(3)
ds_write_b128 v6, v[32:35] offset:16384 // (op 13) Store local data 
// VMEM: Expected complete at 203 (current 183)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
s_waitcnt vmcnt(2)
ds_write_b128 v6, v[36:39] offset:16640 // (op 13) Store local data 
// VMEM: Expected complete at 205 (current 185)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
s_waitcnt vmcnt(1)
ds_write_b128 v6, v[40:43] offset:16896 // (op 13) Store local data 
// VMEM: Expected complete at 207 (current 187)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
s_waitcnt vmcnt(0)
ds_write_b128 v6, v[44:47] offset:17152 // (op 13) Store local data 
// DSMEM: Expected stall of 14. CBNW: 0, Inc: 4
// VMEM: Expected complete at 223 (current 203)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 41) StoreLDSTile Value: Float(13) END
 // (op 41) Deallocate{}(654) BEGIN
 // (op 654) Deallocate 7
 // Freeing : 16 VGPRs (Value: Float) (op 10): v32, v33, v34, v35, v36, v37, v38, v39, v40, v41, v42, v43, v44, v45, v46, v47

 // Deleting tag 7
 // (op 41) Deallocate{}(654) END
 // (op 41) Barrier(542) BEGIN
s_waitcnt lgkmcnt(0)
s_barrier  // (op 542) 
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 41) Barrier(542) END
 // (op 41) SetCoordinate(54) BEGIN
 // (op 54) SetCoordinate (54): Coordinate 282 = 0:U32
 // (op 54) generate({})
 // (op 54) end: generate({})
 // (op 54) generate({56})
 // (op 54) LoadLDSTile{Value: Float}(56) BEGIN
 // (op 56) GEN: loadMacroTileWAVELDS OP 56 LDS 22 WaveTile 147
 // (op 56) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 56) Generate 256:U32 into nullptr
 // tag 20: v**UNALLOCATED**
 // (op 56) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 56) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 56): v10
ds_read_b32 v10, v4 offset:16384 // (op 56) Load local data 
// VMEM: Expected complete at 246 (current 226)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 54) LoadLDSTile{Value: Float}(56) END
 // (op 54) end: generate({56})
 // (op 41) SetCoordinate(54) END
 // (op 41) Multiply(59) BEGIN
 // (op 59) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 59) 
 // (op 41) Multiply(59) END
 // (op 41) Deallocate{}(660) BEGIN
 // (op 660) Deallocate 20
 // Freeing : 1 VGPR (Value: Float) (op 56): v10

 // Deleting tag 20
 // (op 660) Deallocate 12
 // Freeing : 1 VGPR (Value: Float) (op 51): v11

 // Deleting tag 12
 // (op 41) Deallocate{}(660) END
 // (op 41) SetCoordinate(60) BEGIN
 // (op 60) SetCoordinate (60): Coordinate 282 = 1:U32
 // (op 60) generate({})
 // (op 60) end: generate({})
 // (op 60) generate({62})
 // (op 60) LoadLDSTile{Value: Float}(62) BEGIN
 // (op 62) GEN: loadMacroTileWAVELDS OP 62 LDS 14 WaveTile 87
 // (op 62) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 62) Generate 256:U32 into nullptr
 // tag 285: v**UNALLOCATED**
 // (op 62) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 62) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 512
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 62): v11
ds_read_b32 v11, v0 offset:512 // (op 62) Load local data 
// VMEM: Expected complete at 269 (current 249)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 60) LoadLDSTile{Value: Float}(62) END
 // (op 60) end: generate({62})
 // (op 41) SetCoordinate(60) END
 // (op 41) SetCoordinate(64) BEGIN
 // (op 64) SetCoordinate (64): Coordinate 282 = 1:U32
 // (op 64) generate({})
 // (op 64) end: generate({})
 // (op 64) generate({66})
 // (op 64) LoadLDSTile{Value: Float}(66) BEGIN
 // (op 66) GEN: loadMacroTileWAVELDS OP 66 LDS 22 WaveTile 147
 // (op 66) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 66) Generate 256:U32 into nullptr
 // tag 287: v**UNALLOCATED**
 // (op 66) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 66) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16896
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 66): v10
ds_read_b32 v10, v4 offset:16896 // (op 66) Load local data 
// VMEM: Expected complete at 270 (current 250)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 64) LoadLDSTile{Value: Float}(66) END
 // (op 64) end: generate({66})
 // (op 41) SetCoordinate(64) END
 // (op 41) Multiply(68) BEGIN
 // (op 68) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 68) 
 // (op 41) Multiply(68) END
 // (op 41) Deallocate{}(662) BEGIN
 // (op 662) Deallocate 287
 // Freeing : 1 VGPR (Value: Float) (op 66): v10

 // Deleting tag 287
 // (op 662) Deallocate 285
 // Freeing : 1 VGPR (Value: Float) (op 62): v11

 // Deleting tag 285
 // (op 41) Deallocate{}(662) END
 // (op 41) SetCoordinate(69) BEGIN
 // (op 69) SetCoordinate (69): Coordinate 282 = 2:U32
 // (op 69) generate({})
 // (op 69) end: generate({})
 // (op 69) generate({71})
 // (op 69) LoadLDSTile{Value: Float}(71) BEGIN
 // (op 71) GEN: loadMacroTileWAVELDS OP 71 LDS 14 WaveTile 87
 // (op 71) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 71) Generate 256:U32 into nullptr
 // tag 289: v**UNALLOCATED**
 // (op 71) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 71) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
// Allocated : 1 VGPR (Value: Float) (op 71): v11
ds_read_b32 v11, v0 offset:1024 // (op 71) Load local data 
// VMEM: Expected complete at 307 (current 287)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 69) LoadLDSTile{Value: Float}(71) END
 // (op 69) end: generate({71})
 // (op 41) SetCoordinate(69) END
 // (op 41) SetCoordinate(73) BEGIN
 // (op 73) SetCoordinate (73): Coordinate 282 = 2:U32
 // (op 73) generate({})
 // (op 73) end: generate({})
 // (op 73) generate({75})
 // (op 73) LoadLDSTile{Value: Float}(75) BEGIN
 // (op 75) GEN: loadMacroTileWAVELDS OP 75 LDS 22 WaveTile 147
 // (op 75) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 75) Generate 256:U32 into nullptr
 // tag 291: v**UNALLOCATED**
 // (op 75) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 75) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 17408
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 75): v10
ds_read_b32 v10, v4 offset:17408 // (op 75) Load local data 
// VMEM: Expected complete at 308 (current 288)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 73) LoadLDSTile{Value: Float}(75) END
 // (op 73) end: generate({75})
 // (op 41) SetCoordinate(73) END
 // (op 41) Multiply(77) BEGIN
 // (op 77) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 77) 
 // (op 41) Multiply(77) END
 // (op 41) Deallocate{}(664) BEGIN
 // (op 664) Deallocate 291
 // Freeing : 1 VGPR (Value: Float) (op 75): v10

 // Deleting tag 291
 // (op 664) Deallocate 289
 // Freeing : 1 VGPR (Value: Float) (op 71): v11

 // Deleting tag 289
 // (op 41) Deallocate{}(664) END
 // (op 41) SetCoordinate(78) BEGIN
 // (op 78) SetCoordinate (78): Coordinate 282 = 3:U32
 // (op 78) generate({})
 // (op 78) end: generate({})
 // (op 78) generate({80})
 // (op 78) LoadLDSTile{Value: Float}(80) BEGIN
 // (op 80) GEN: loadMacroTileWAVELDS OP 80 LDS 14 WaveTile 87
 // (op 80) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 80) Generate 256:U32 into nullptr
 // tag 293: v**UNALLOCATED**
 // (op 80) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 80) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
// Allocated : 1 VGPR (Value: Float) (op 80): v11
ds_read_b32 v11, v0 offset:1536 // (op 80) Load local data 
// VMEM: Expected complete at 345 (current 325)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 78) LoadLDSTile{Value: Float}(80) END
 // (op 78) end: generate({80})
 // (op 41) SetCoordinate(78) END
 // (op 41) SetCoordinate(82) BEGIN
 // (op 82) SetCoordinate (82): Coordinate 282 = 3:U32
 // (op 82) generate({})
 // (op 82) end: generate({})
 // (op 82) generate({84})
 // (op 82) LoadLDSTile{Value: Float}(84) BEGIN
 // (op 84) GEN: loadMacroTileWAVELDS OP 84 LDS 22 WaveTile 147
 // (op 84) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 84) Generate 256:U32 into nullptr
 // tag 295: v**UNALLOCATED**
 // (op 84) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 84) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 17920
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 84): v10
ds_read_b32 v10, v4 offset:17920 // (op 84) Load local data 
// VMEM: Expected complete at 346 (current 326)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 82) LoadLDSTile{Value: Float}(84) END
 // (op 82) end: generate({84})
 // (op 41) SetCoordinate(82) END
 // (op 41) Multiply(86) BEGIN
 // (op 86) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 86) 
 // (op 41) Multiply(86) END
 // (op 41) Deallocate{}(666) BEGIN
 // (op 666) Deallocate 295
 // Freeing : 1 VGPR (Value: Float) (op 84): v10

 // Deleting tag 295
 // (op 666) Deallocate 293
 // Freeing : 1 VGPR (Value: Float) (op 80): v11

 // Deleting tag 293
 // (op 41) Deallocate{}(666) END
 // (op 41) SetCoordinate(87) BEGIN
 // (op 87) SetCoordinate (87): Coordinate 282 = 4:U32
 // (op 87) generate({})
 // (op 87) end: generate({})
 // (op 87) generate({89})
 // (op 87) LoadLDSTile{Value: Float}(89) BEGIN
 // (op 89) GEN: loadMacroTileWAVELDS OP 89 LDS 14 WaveTile 87
 // (op 89) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 89) Generate 256:U32 into nullptr
 // tag 297: v**UNALLOCATED**
 // (op 89) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 89) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
// Allocated : 1 VGPR (Value: Float) (op 89): v11
ds_read_b32 v11, v0 offset:2048 // (op 89) Load local data 
// VMEM: Expected complete at 383 (current 363)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 87) LoadLDSTile{Value: Float}(89) END
 // (op 87) end: generate({89})
 // (op 41) SetCoordinate(87) END
 // (op 41) SetCoordinate(91) BEGIN
 // (op 91) SetCoordinate (91): Coordinate 282 = 4:U32
 // (op 91) generate({})
 // (op 91) end: generate({})
 // (op 91) generate({93})
 // (op 91) LoadLDSTile{Value: Float}(93) BEGIN
 // (op 93) GEN: loadMacroTileWAVELDS OP 93 LDS 22 WaveTile 147
 // (op 93) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 93) Generate 256:U32 into nullptr
 // tag 299: v**UNALLOCATED**
 // (op 93) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 93) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 18432
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 93): v10
ds_read_b32 v10, v4 offset:18432 // (op 93) Load local data 
// VMEM: Expected complete at 384 (current 364)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 91) LoadLDSTile{Value: Float}(93) END
 // (op 91) end: generate({93})
 // (op 41) SetCoordinate(91) END
 // (op 41) Multiply(95) BEGIN
 // (op 95) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 95) 
 // (op 41) Multiply(95) END
 // (op 41) Deallocate{}(668) BEGIN
 // (op 668) Deallocate 297
 // Freeing : 1 VGPR (Value: Float) (op 89): v11

 // Deleting tag 297
 // (op 668) Deallocate 299
 // Freeing : 1 VGPR (Value: Float) (op 93): v10

 // Deleting tag 299
 // (op 41) Deallocate{}(668) END
 // (op 41) SetCoordinate(96) BEGIN
 // (op 96) SetCoordinate (96): Coordinate 282 = 5:U32
 // (op 96) generate({})
 // (op 96) end: generate({})
 // (op 96) generate({98})
 // (op 96) LoadLDSTile{Value: Float}(98) BEGIN
 // (op 98) GEN: loadMacroTileWAVELDS OP 98 LDS 14 WaveTile 87
 // (op 98) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 98) Generate 256:U32 into nullptr
 // tag 301: v**UNALLOCATED**
 // (op 98) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 98) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 2560
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 98): v11
ds_read_b32 v11, v0 offset:2560 // (op 98) Load local data 
// VMEM: Expected complete at 421 (current 401)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 96) LoadLDSTile{Value: Float}(98) END
 // (op 96) end: generate({98})
 // (op 41) SetCoordinate(96) END
 // (op 41) SetCoordinate(100) BEGIN
 // (op 100) SetCoordinate (100): Coordinate 282 = 5:U32
 // (op 100) generate({})
 // (op 100) end: generate({})
 // (op 100) generate({102})
 // (op 100) LoadLDSTile{Value: Float}(102) BEGIN
 // (op 102) GEN: loadMacroTileWAVELDS OP 102 LDS 22 WaveTile 147
 // (op 102) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 102) Generate 256:U32 into nullptr
 // tag 303: v**UNALLOCATED**
 // (op 102) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 102) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 18944
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 102): v10
ds_read_b32 v10, v4 offset:18944 // (op 102) Load local data 
// VMEM: Expected complete at 422 (current 402)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 100) LoadLDSTile{Value: Float}(102) END
 // (op 100) end: generate({102})
 // (op 41) SetCoordinate(100) END
 // (op 41) Multiply(104) BEGIN
 // (op 104) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 104) 
 // (op 41) Multiply(104) END
 // (op 41) Deallocate{}(670) BEGIN
 // (op 670) Deallocate 301
 // Freeing : 1 VGPR (Value: Float) (op 98): v11

 // Deleting tag 301
 // (op 670) Deallocate 303
 // Freeing : 1 VGPR (Value: Float) (op 102): v10

 // Deleting tag 303
 // (op 41) Deallocate{}(670) END
 // (op 41) SetCoordinate(105) BEGIN
 // (op 105) SetCoordinate (105): Coordinate 282 = 6:U32
 // (op 105) generate({})
 // (op 105) end: generate({})
 // (op 105) generate({107})
 // (op 105) LoadLDSTile{Value: Float}(107) BEGIN
 // (op 107) GEN: loadMacroTileWAVELDS OP 107 LDS 14 WaveTile 87
 // (op 107) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 107) Generate 256:U32 into nullptr
 // tag 305: v**UNALLOCATED**
 // (op 107) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 107) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 3072
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 107): v11
ds_read_b32 v11, v0 offset:3072 // (op 107) Load local data 
// VMEM: Expected complete at 459 (current 439)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 105) LoadLDSTile{Value: Float}(107) END
 // (op 105) end: generate({107})
 // (op 41) SetCoordinate(105) END
 // (op 41) SetCoordinate(109) BEGIN
 // (op 109) SetCoordinate (109): Coordinate 282 = 6:U32
 // (op 109) generate({})
 // (op 109) end: generate({})
 // (op 109) generate({111})
 // (op 109) LoadLDSTile{Value: Float}(111) BEGIN
 // (op 111) GEN: loadMacroTileWAVELDS OP 111 LDS 22 WaveTile 147
 // (op 111) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 111) Generate 256:U32 into nullptr
 // tag 307: v**UNALLOCATED**
 // (op 111) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 111) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 19456
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 111): v10
ds_read_b32 v10, v4 offset:19456 // (op 111) Load local data 
// VMEM: Expected complete at 460 (current 440)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 109) LoadLDSTile{Value: Float}(111) END
 // (op 109) end: generate({111})
 // (op 41) SetCoordinate(109) END
 // (op 41) Multiply(113) BEGIN
 // (op 113) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 113) 
 // (op 41) Multiply(113) END
 // (op 41) Deallocate{}(672) BEGIN
 // (op 672) Deallocate 305
 // Freeing : 1 VGPR (Value: Float) (op 107): v11

 // Deleting tag 305
 // (op 672) Deallocate 307
 // Freeing : 1 VGPR (Value: Float) (op 111): v10

 // Deleting tag 307
 // (op 41) Deallocate{}(672) END
 // (op 41) SetCoordinate(114) BEGIN
 // (op 114) SetCoordinate (114): Coordinate 282 = 7:U32
 // (op 114) generate({})
 // (op 114) end: generate({})
 // (op 114) generate({116})
 // (op 114) LoadLDSTile{Value: Float}(116) BEGIN
 // (op 116) GEN: loadMacroTileWAVELDS OP 116 LDS 14 WaveTile 87
 // (op 116) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 116) Generate 256:U32 into nullptr
 // tag 309: v**UNALLOCATED**
 // (op 116) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 116) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 3584
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 116): v11
ds_read_b32 v11, v0 offset:3584 // (op 116) Load local data 
// VMEM: Expected complete at 497 (current 477)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 114) LoadLDSTile{Value: Float}(116) END
 // (op 114) end: generate({116})
 // (op 41) SetCoordinate(114) END
 // (op 41) SetCoordinate(118) BEGIN
 // (op 118) SetCoordinate (118): Coordinate 282 = 7:U32
 // (op 118) generate({})
 // (op 118) end: generate({})
 // (op 118) generate({120})
 // (op 118) LoadLDSTile{Value: Float}(120) BEGIN
 // (op 120) GEN: loadMacroTileWAVELDS OP 120 LDS 22 WaveTile 147
 // (op 120) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 120) Generate 256:U32 into nullptr
 // tag 311: v**UNALLOCATED**
 // (op 120) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 120) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 19968
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 120): v10
ds_read_b32 v10, v4 offset:19968 // (op 120) Load local data 
// VMEM: Expected complete at 498 (current 478)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 118) LoadLDSTile{Value: Float}(120) END
 // (op 118) end: generate({120})
 // (op 41) SetCoordinate(118) END
 // (op 41) Multiply(122) BEGIN
 // (op 122) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 122) 
 // (op 41) Multiply(122) END
 // (op 41) Deallocate{}(674) BEGIN
 // (op 674) Deallocate 309
 // Freeing : 1 VGPR (Value: Float) (op 116): v11

 // Deleting tag 309
 // (op 674) Deallocate 311
 // Freeing : 1 VGPR (Value: Float) (op 120): v10

 // Deleting tag 311
 // (op 41) Deallocate{}(674) END
 // (op 41) SetCoordinate(123) BEGIN
 // (op 123) SetCoordinate (123): Coordinate 282 = 8:U32
 // (op 123) generate({})
 // (op 123) end: generate({})
 // (op 123) generate({125})
 // (op 123) LoadLDSTile{Value: Float}(125) BEGIN
 // (op 125) GEN: loadMacroTileWAVELDS OP 125 LDS 14 WaveTile 87
 // (op 125) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 125) Generate 256:U32 into nullptr
 // tag 313: v**UNALLOCATED**
 // (op 125) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 125) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
// Allocated : 1 VGPR (Value: Float) (op 125): v11
ds_read_b32 v11, v0 offset:4096 // (op 125) Load local data 
// VMEM: Expected complete at 535 (current 515)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 123) LoadLDSTile{Value: Float}(125) END
 // (op 123) end: generate({125})
 // (op 41) SetCoordinate(123) END
 // (op 41) SetCoordinate(127) BEGIN
 // (op 127) SetCoordinate (127): Coordinate 282 = 8:U32
 // (op 127) generate({})
 // (op 127) end: generate({})
 // (op 127) generate({129})
 // (op 127) LoadLDSTile{Value: Float}(129) BEGIN
 // (op 129) GEN: loadMacroTileWAVELDS OP 129 LDS 22 WaveTile 147
 // (op 129) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 129) Generate 256:U32 into nullptr
 // tag 315: v**UNALLOCATED**
 // (op 129) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 129) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 20480
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 129): v10
ds_read_b32 v10, v4 offset:20480 // (op 129) Load local data 
// VMEM: Expected complete at 536 (current 516)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 127) LoadLDSTile{Value: Float}(129) END
 // (op 127) end: generate({129})
 // (op 41) SetCoordinate(127) END
 // (op 41) Multiply(131) BEGIN
 // (op 131) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 131) 
 // (op 41) Multiply(131) END
 // (op 41) Deallocate{}(676) BEGIN
 // (op 676) Deallocate 315
 // Freeing : 1 VGPR (Value: Float) (op 129): v10

 // Deleting tag 315
 // (op 676) Deallocate 313
 // Freeing : 1 VGPR (Value: Float) (op 125): v11

 // Deleting tag 313
 // (op 41) Deallocate{}(676) END
 // (op 41) SetCoordinate(132) BEGIN
 // (op 132) SetCoordinate (132): Coordinate 282 = 9:U32
 // (op 132) generate({})
 // (op 132) end: generate({})
 // (op 132) generate({134})
 // (op 132) LoadLDSTile{Value: Float}(134) BEGIN
 // (op 134) GEN: loadMacroTileWAVELDS OP 134 LDS 14 WaveTile 87
 // (op 134) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 134) Generate 256:U32 into nullptr
 // tag 317: v**UNALLOCATED**
 // (op 134) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 134) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 4608
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 134): v11
ds_read_b32 v11, v0 offset:4608 // (op 134) Load local data 
// VMEM: Expected complete at 573 (current 553)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 132) LoadLDSTile{Value: Float}(134) END
 // (op 132) end: generate({134})
 // (op 41) SetCoordinate(132) END
 // (op 41) SetCoordinate(136) BEGIN
 // (op 136) SetCoordinate (136): Coordinate 282 = 9:U32
 // (op 136) generate({})
 // (op 136) end: generate({})
 // (op 136) generate({138})
 // (op 136) LoadLDSTile{Value: Float}(138) BEGIN
 // (op 138) GEN: loadMacroTileWAVELDS OP 138 LDS 22 WaveTile 147
 // (op 138) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 138) Generate 256:U32 into nullptr
 // tag 319: v**UNALLOCATED**
 // (op 138) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 138) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 20992
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 138): v10
ds_read_b32 v10, v4 offset:20992 // (op 138) Load local data 
// VMEM: Expected complete at 574 (current 554)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 136) LoadLDSTile{Value: Float}(138) END
 // (op 136) end: generate({138})
 // (op 41) SetCoordinate(136) END
 // (op 41) Multiply(140) BEGIN
 // (op 140) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 140) 
 // (op 41) Multiply(140) END
 // (op 41) Deallocate{}(678) BEGIN
 // (op 678) Deallocate 319
 // Freeing : 1 VGPR (Value: Float) (op 138): v10

 // Deleting tag 319
 // (op 678) Deallocate 317
 // Freeing : 1 VGPR (Value: Float) (op 134): v11

 // Deleting tag 317
 // (op 41) Deallocate{}(678) END
 // (op 41) SetCoordinate(141) BEGIN
 // (op 141) SetCoordinate (141): Coordinate 282 = 10:U32
 // (op 141) generate({})
 // (op 141) end: generate({})
 // (op 141) generate({143})
 // (op 141) LoadLDSTile{Value: Float}(143) BEGIN
 // (op 143) GEN: loadMacroTileWAVELDS OP 143 LDS 14 WaveTile 87
 // (op 143) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 143) Generate 256:U32 into nullptr
 // tag 321: v**UNALLOCATED**
 // (op 143) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 143) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 5120
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 143): v11
ds_read_b32 v11, v0 offset:5120 // (op 143) Load local data 
// VMEM: Expected complete at 611 (current 591)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 141) LoadLDSTile{Value: Float}(143) END
 // (op 141) end: generate({143})
 // (op 41) SetCoordinate(141) END
 // (op 41) SetCoordinate(145) BEGIN
 // (op 145) SetCoordinate (145): Coordinate 282 = 10:U32
 // (op 145) generate({})
 // (op 145) end: generate({})
 // (op 145) generate({147})
 // (op 145) LoadLDSTile{Value: Float}(147) BEGIN
 // (op 147) GEN: loadMacroTileWAVELDS OP 147 LDS 22 WaveTile 147
 // (op 147) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 147) Generate 256:U32 into nullptr
 // tag 323: v**UNALLOCATED**
 // (op 147) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 147) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 21504
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 147): v10
ds_read_b32 v10, v4 offset:21504 // (op 147) Load local data 
// VMEM: Expected complete at 612 (current 592)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 145) LoadLDSTile{Value: Float}(147) END
 // (op 145) end: generate({147})
 // (op 41) SetCoordinate(145) END
 // (op 41) Multiply(149) BEGIN
 // (op 149) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 149) 
 // (op 41) Multiply(149) END
 // (op 41) Deallocate{}(680) BEGIN
 // (op 680) Deallocate 323
 // Freeing : 1 VGPR (Value: Float) (op 147): v10

 // Deleting tag 323
 // (op 680) Deallocate 321
 // Freeing : 1 VGPR (Value: Float) (op 143): v11

 // Deleting tag 321
 // (op 41) Deallocate{}(680) END
 // (op 41) SetCoordinate(150) BEGIN
 // (op 150) SetCoordinate (150): Coordinate 282 = 11:U32
 // (op 150) generate({})
 // (op 150) end: generate({})
 // (op 150) generate({152})
 // (op 150) LoadLDSTile{Value: Float}(152) BEGIN
 // (op 152) GEN: loadMacroTileWAVELDS OP 152 LDS 14 WaveTile 87
 // (op 152) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 152) Generate 256:U32 into nullptr
 // tag 325: v**UNALLOCATED**
 // (op 152) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 152) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
// Allocated : 1 VGPR (Value: Float) (op 152): v11
ds_read_b32 v11, v0 offset:5632 // (op 152) Load local data 
// VMEM: Expected complete at 649 (current 629)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 150) LoadLDSTile{Value: Float}(152) END
 // (op 150) end: generate({152})
 // (op 41) SetCoordinate(150) END
 // (op 41) SetCoordinate(154) BEGIN
 // (op 154) SetCoordinate (154): Coordinate 282 = 11:U32
 // (op 154) generate({})
 // (op 154) end: generate({})
 // (op 154) generate({156})
 // (op 154) LoadLDSTile{Value: Float}(156) BEGIN
 // (op 156) GEN: loadMacroTileWAVELDS OP 156 LDS 22 WaveTile 147
 // (op 156) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 156) Generate 256:U32 into nullptr
 // tag 327: v**UNALLOCATED**
 // (op 156) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 156) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 22016
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 156): v10
ds_read_b32 v10, v4 offset:22016 // (op 156) Load local data 
// VMEM: Expected complete at 650 (current 630)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 154) LoadLDSTile{Value: Float}(156) END
 // (op 154) end: generate({156})
 // (op 41) SetCoordinate(154) END
 // (op 41) Multiply(158) BEGIN
 // (op 158) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 158) 
 // (op 41) Multiply(158) END
 // (op 41) Deallocate{}(682) BEGIN
 // (op 682) Deallocate 327
 // Freeing : 1 VGPR (Value: Float) (op 156): v10

 // Deleting tag 327
 // (op 682) Deallocate 325
 // Freeing : 1 VGPR (Value: Float) (op 152): v11

 // Deleting tag 325
 // (op 41) Deallocate{}(682) END
 // (op 41) SetCoordinate(159) BEGIN
 // (op 159) SetCoordinate (159): Coordinate 282 = 12:U32
 // (op 159) generate({})
 // (op 159) end: generate({})
 // (op 159) generate({161})
 // (op 159) LoadLDSTile{Value: Float}(161) BEGIN
 // (op 161) GEN: loadMacroTileWAVELDS OP 161 LDS 14 WaveTile 87
 // (op 161) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 161) Generate 256:U32 into nullptr
 // tag 329: v**UNALLOCATED**
 // (op 161) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 161) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
// Allocated : 1 VGPR (Value: Float) (op 161): v11
ds_read_b32 v11, v0 offset:6144 // (op 161) Load local data 
// VMEM: Expected complete at 687 (current 667)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 159) LoadLDSTile{Value: Float}(161) END
 // (op 159) end: generate({161})
 // (op 41) SetCoordinate(159) END
 // (op 41) SetCoordinate(163) BEGIN
 // (op 163) SetCoordinate (163): Coordinate 282 = 12:U32
 // (op 163) generate({})
 // (op 163) end: generate({})
 // (op 163) generate({165})
 // (op 163) LoadLDSTile{Value: Float}(165) BEGIN
 // (op 165) GEN: loadMacroTileWAVELDS OP 165 LDS 22 WaveTile 147
 // (op 165) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 165) Generate 256:U32 into nullptr
 // tag 331: v**UNALLOCATED**
 // (op 165) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 165) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 22528
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 165): v10
ds_read_b32 v10, v4 offset:22528 // (op 165) Load local data 
// VMEM: Expected complete at 688 (current 668)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 163) LoadLDSTile{Value: Float}(165) END
 // (op 163) end: generate({165})
 // (op 41) SetCoordinate(163) END
 // (op 41) Multiply(167) BEGIN
 // (op 167) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 167) 
 // (op 41) Multiply(167) END
 // (op 41) Deallocate{}(684) BEGIN
 // (op 684) Deallocate 331
 // Freeing : 1 VGPR (Value: Float) (op 165): v10

 // Deleting tag 331
 // (op 684) Deallocate 329
 // Freeing : 1 VGPR (Value: Float) (op 161): v11

 // Deleting tag 329
 // (op 41) Deallocate{}(684) END
 // (op 41) SetCoordinate(168) BEGIN
 // (op 168) SetCoordinate (168): Coordinate 282 = 13:U32
 // (op 168) generate({})
 // (op 168) end: generate({})
 // (op 168) generate({170})
 // (op 168) LoadLDSTile{Value: Float}(170) BEGIN
 // (op 170) GEN: loadMacroTileWAVELDS OP 170 LDS 14 WaveTile 87
 // (op 170) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 170) Generate 256:U32 into nullptr
 // tag 333: v**UNALLOCATED**
 // (op 170) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 170) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 6656
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 170): v11
ds_read_b32 v11, v0 offset:6656 // (op 170) Load local data 
// VMEM: Expected complete at 725 (current 705)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 168) LoadLDSTile{Value: Float}(170) END
 // (op 168) end: generate({170})
 // (op 41) SetCoordinate(168) END
 // (op 41) SetCoordinate(172) BEGIN
 // (op 172) SetCoordinate (172): Coordinate 282 = 13:U32
 // (op 172) generate({})
 // (op 172) end: generate({})
 // (op 172) generate({174})
 // (op 172) LoadLDSTile{Value: Float}(174) BEGIN
 // (op 174) GEN: loadMacroTileWAVELDS OP 174 LDS 22 WaveTile 147
 // (op 174) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 174) Generate 256:U32 into nullptr
 // tag 335: v**UNALLOCATED**
 // (op 174) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 174) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 23040
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 174): v10
ds_read_b32 v10, v4 offset:23040 // (op 174) Load local data 
// VMEM: Expected complete at 726 (current 706)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 172) LoadLDSTile{Value: Float}(174) END
 // (op 172) end: generate({174})
 // (op 41) SetCoordinate(172) END
 // (op 41) Multiply(176) BEGIN
 // (op 176) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 176) 
 // (op 41) Multiply(176) END
 // (op 41) Deallocate{}(686) BEGIN
 // (op 686) Deallocate 333
 // Freeing : 1 VGPR (Value: Float) (op 170): v11

 // Deleting tag 333
 // (op 686) Deallocate 335
 // Freeing : 1 VGPR (Value: Float) (op 174): v10

 // Deleting tag 335
 // (op 41) Deallocate{}(686) END
 // (op 41) SetCoordinate(177) BEGIN
 // (op 177) SetCoordinate (177): Coordinate 282 = 14:U32
 // (op 177) generate({})
 // (op 177) end: generate({})
 // (op 177) generate({179})
 // (op 177) LoadLDSTile{Value: Float}(179) BEGIN
 // (op 179) GEN: loadMacroTileWAVELDS OP 179 LDS 14 WaveTile 87
 // (op 179) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 179) Generate 256:U32 into nullptr
 // tag 337: v**UNALLOCATED**
 // (op 179) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 179) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7168
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 179): v11
ds_read_b32 v11, v0 offset:7168 // (op 179) Load local data 
// VMEM: Expected complete at 763 (current 743)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 177) LoadLDSTile{Value: Float}(179) END
 // (op 177) end: generate({179})
 // (op 41) SetCoordinate(177) END
 // (op 41) SetCoordinate(181) BEGIN
 // (op 181) SetCoordinate (181): Coordinate 282 = 14:U32
 // (op 181) generate({})
 // (op 181) end: generate({})
 // (op 181) generate({183})
 // (op 181) LoadLDSTile{Value: Float}(183) BEGIN
 // (op 183) GEN: loadMacroTileWAVELDS OP 183 LDS 22 WaveTile 147
 // (op 183) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 183) Generate 256:U32 into nullptr
 // tag 339: v**UNALLOCATED**
 // (op 183) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 183) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 23552
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 183): v10
ds_read_b32 v10, v4 offset:23552 // (op 183) Load local data 
// VMEM: Expected complete at 764 (current 744)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 181) LoadLDSTile{Value: Float}(183) END
 // (op 181) end: generate({183})
 // (op 41) SetCoordinate(181) END
 // (op 41) Multiply(185) BEGIN
 // (op 185) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 185) 
 // (op 41) Multiply(185) END
 // (op 41) Deallocate{}(688) BEGIN
 // (op 688) Deallocate 337
 // Freeing : 1 VGPR (Value: Float) (op 179): v11

 // Deleting tag 337
 // (op 688) Deallocate 339
 // Freeing : 1 VGPR (Value: Float) (op 183): v10

 // Deleting tag 339
 // (op 41) Deallocate{}(688) END
 // (op 41) SetCoordinate(186) BEGIN
 // (op 186) SetCoordinate (186): Coordinate 282 = 15:U32
 // (op 186) generate({})
 // (op 186) end: generate({})
 // (op 186) generate({188})
 // (op 186) LoadLDSTile{Value: Float}(188) BEGIN
 // (op 188) GEN: loadMacroTileWAVELDS OP 188 LDS 14 WaveTile 87
 // (op 188) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 188) Generate 256:U32 into nullptr
 // tag 341: v**UNALLOCATED**
 // (op 188) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 188) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 7680
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 188): v11
ds_read_b32 v11, v0 offset:7680 // (op 188) Load local data 
// VMEM: Expected complete at 801 (current 781)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 186) LoadLDSTile{Value: Float}(188) END
 // (op 186) end: generate({188})
 // (op 41) SetCoordinate(186) END
 // (op 41) SetCoordinate(190) BEGIN
 // (op 190) SetCoordinate (190): Coordinate 282 = 15:U32
 // (op 190) generate({})
 // (op 190) end: generate({})
 // (op 190) generate({192})
 // (op 190) LoadLDSTile{Value: Float}(192) BEGIN
 // (op 192) GEN: loadMacroTileWAVELDS OP 192 LDS 22 WaveTile 147
 // (op 192) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 192) Generate 256:U32 into nullptr
 // tag 343: v**UNALLOCATED**
 // (op 192) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 192) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 24064
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 192): v10
ds_read_b32 v10, v4 offset:24064 // (op 192) Load local data 
// VMEM: Expected complete at 802 (current 782)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 190) LoadLDSTile{Value: Float}(192) END
 // (op 190) end: generate({192})
 // (op 41) SetCoordinate(190) END
 // (op 41) Multiply(194) BEGIN
 // (op 194) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 194) 
 // (op 41) Multiply(194) END
 // (op 41) Deallocate{}(690) BEGIN
 // (op 690) Deallocate 341
 // Freeing : 1 VGPR (Value: Float) (op 188): v11

 // Deleting tag 341
 // (op 690) Deallocate 343
 // Freeing : 1 VGPR (Value: Float) (op 192): v10

 // Deleting tag 343
 // (op 41) Deallocate{}(690) END
 // (op 41) SetCoordinate(195) BEGIN
 // (op 195) SetCoordinate (195): Coordinate 282 = 16:U32
 // (op 195) generate({})
 // (op 195) end: generate({})
 // (op 195) generate({197})
 // (op 195) LoadLDSTile{Value: Float}(197) BEGIN
 // (op 197) GEN: loadMacroTileWAVELDS OP 197 LDS 14 WaveTile 87
 // (op 197) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 197) Generate 256:U32 into nullptr
 // tag 345: v**UNALLOCATED**
 // (op 197) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 197) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
// Allocated : 1 VGPR (Value: Float) (op 197): v11
ds_read_b32 v11, v0 offset:8192 // (op 197) Load local data 
// VMEM: Expected complete at 839 (current 819)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 195) LoadLDSTile{Value: Float}(197) END
 // (op 195) end: generate({197})
 // (op 41) SetCoordinate(195) END
 // (op 41) SetCoordinate(199) BEGIN
 // (op 199) SetCoordinate (199): Coordinate 282 = 16:U32
 // (op 199) generate({})
 // (op 199) end: generate({})
 // (op 199) generate({201})
 // (op 199) LoadLDSTile{Value: Float}(201) BEGIN
 // (op 201) GEN: loadMacroTileWAVELDS OP 201 LDS 22 WaveTile 147
 // (op 201) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 201) Generate 256:U32 into nullptr
 // tag 347: v**UNALLOCATED**
 // (op 201) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 201) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 24576
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 201): v10
ds_read_b32 v10, v4 offset:24576 // (op 201) Load local data 
// VMEM: Expected complete at 840 (current 820)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 199) LoadLDSTile{Value: Float}(201) END
 // (op 199) end: generate({201})
 // (op 41) SetCoordinate(199) END
 // (op 41) Multiply(203) BEGIN
 // (op 203) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 203) 
 // (op 41) Multiply(203) END
 // (op 41) Deallocate{}(692) BEGIN
 // (op 692) Deallocate 345
 // Freeing : 1 VGPR (Value: Float) (op 197): v11

 // Deleting tag 345
 // (op 692) Deallocate 347
 // Freeing : 1 VGPR (Value: Float) (op 201): v10

 // Deleting tag 347
 // (op 41) Deallocate{}(692) END
 // (op 41) SetCoordinate(204) BEGIN
 // (op 204) SetCoordinate (204): Coordinate 282 = 17:U32
 // (op 204) generate({})
 // (op 204) end: generate({})
 // (op 204) generate({206})
 // (op 204) LoadLDSTile{Value: Float}(206) BEGIN
 // (op 206) GEN: loadMacroTileWAVELDS OP 206 LDS 14 WaveTile 87
 // (op 206) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 206) Generate 256:U32 into nullptr
 // tag 349: v**UNALLOCATED**
 // (op 206) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 206) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 8704
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 206): v11
ds_read_b32 v11, v0 offset:8704 // (op 206) Load local data 
// VMEM: Expected complete at 877 (current 857)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 204) LoadLDSTile{Value: Float}(206) END
 // (op 204) end: generate({206})
 // (op 41) SetCoordinate(204) END
 // (op 41) SetCoordinate(208) BEGIN
 // (op 208) SetCoordinate (208): Coordinate 282 = 17:U32
 // (op 208) generate({})
 // (op 208) end: generate({})
 // (op 208) generate({210})
 // (op 208) LoadLDSTile{Value: Float}(210) BEGIN
 // (op 210) GEN: loadMacroTileWAVELDS OP 210 LDS 22 WaveTile 147
 // (op 210) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 210) Generate 256:U32 into nullptr
 // tag 351: v**UNALLOCATED**
 // (op 210) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 210) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 25088
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 210): v10
ds_read_b32 v10, v4 offset:25088 // (op 210) Load local data 
// VMEM: Expected complete at 878 (current 858)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 208) LoadLDSTile{Value: Float}(210) END
 // (op 208) end: generate({210})
 // (op 41) SetCoordinate(208) END
 // (op 41) Multiply(212) BEGIN
 // (op 212) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 212) 
 // (op 41) Multiply(212) END
 // (op 41) Deallocate{}(694) BEGIN
 // (op 694) Deallocate 349
 // Freeing : 1 VGPR (Value: Float) (op 206): v11

 // Deleting tag 349
 // (op 694) Deallocate 351
 // Freeing : 1 VGPR (Value: Float) (op 210): v10

 // Deleting tag 351
 // (op 41) Deallocate{}(694) END
 // (op 41) SetCoordinate(213) BEGIN
 // (op 213) SetCoordinate (213): Coordinate 282 = 18:U32
 // (op 213) generate({})
 // (op 213) end: generate({})
 // (op 213) generate({215})
 // (op 213) LoadLDSTile{Value: Float}(215) BEGIN
 // (op 215) GEN: loadMacroTileWAVELDS OP 215 LDS 14 WaveTile 87
 // (op 215) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 215) Generate 256:U32 into nullptr
 // tag 353: v**UNALLOCATED**
 // (op 215) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 215) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 9216
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 215): v11
ds_read_b32 v11, v0 offset:9216 // (op 215) Load local data 
// VMEM: Expected complete at 915 (current 895)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 213) LoadLDSTile{Value: Float}(215) END
 // (op 213) end: generate({215})
 // (op 41) SetCoordinate(213) END
 // (op 41) SetCoordinate(217) BEGIN
 // (op 217) SetCoordinate (217): Coordinate 282 = 18:U32
 // (op 217) generate({})
 // (op 217) end: generate({})
 // (op 217) generate({219})
 // (op 217) LoadLDSTile{Value: Float}(219) BEGIN
 // (op 219) GEN: loadMacroTileWAVELDS OP 219 LDS 22 WaveTile 147
 // (op 219) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 219) Generate 256:U32 into nullptr
 // tag 355: v**UNALLOCATED**
 // (op 219) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 219) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 25600
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 219): v10
ds_read_b32 v10, v4 offset:25600 // (op 219) Load local data 
// VMEM: Expected complete at 916 (current 896)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 217) LoadLDSTile{Value: Float}(219) END
 // (op 217) end: generate({219})
 // (op 41) SetCoordinate(217) END
 // (op 41) Multiply(221) BEGIN
 // (op 221) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 221) 
 // (op 41) Multiply(221) END
 // (op 41) Deallocate{}(696) BEGIN
 // (op 696) Deallocate 353
 // Freeing : 1 VGPR (Value: Float) (op 215): v11

 // Deleting tag 353
 // (op 696) Deallocate 355
 // Freeing : 1 VGPR (Value: Float) (op 219): v10

 // Deleting tag 355
 // (op 41) Deallocate{}(696) END
 // (op 41) SetCoordinate(222) BEGIN
 // (op 222) SetCoordinate (222): Coordinate 282 = 19:U32
 // (op 222) generate({})
 // (op 222) end: generate({})
 // (op 222) generate({224})
 // (op 222) LoadLDSTile{Value: Float}(224) BEGIN
 // (op 224) GEN: loadMacroTileWAVELDS OP 224 LDS 14 WaveTile 87
 // (op 224) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 224) Generate 256:U32 into nullptr
 // tag 357: v**UNALLOCATED**
 // (op 224) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 224) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 9728
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 224): v11
ds_read_b32 v11, v0 offset:9728 // (op 224) Load local data 
// VMEM: Expected complete at 953 (current 933)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 222) LoadLDSTile{Value: Float}(224) END
 // (op 222) end: generate({224})
 // (op 41) SetCoordinate(222) END
 // (op 41) SetCoordinate(226) BEGIN
 // (op 226) SetCoordinate (226): Coordinate 282 = 19:U32
 // (op 226) generate({})
 // (op 226) end: generate({})
 // (op 226) generate({228})
 // (op 226) LoadLDSTile{Value: Float}(228) BEGIN
 // (op 228) GEN: loadMacroTileWAVELDS OP 228 LDS 22 WaveTile 147
 // (op 228) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 228) Generate 256:U32 into nullptr
 // tag 359: v**UNALLOCATED**
 // (op 228) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 228) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 26112
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 228): v10
ds_read_b32 v10, v4 offset:26112 // (op 228) Load local data 
// VMEM: Expected complete at 954 (current 934)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 226) LoadLDSTile{Value: Float}(228) END
 // (op 226) end: generate({228})
 // (op 41) SetCoordinate(226) END
 // (op 41) Multiply(230) BEGIN
 // (op 230) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 230) 
 // (op 41) Multiply(230) END
 // (op 41) Deallocate{}(698) BEGIN
 // (op 698) Deallocate 357
 // Freeing : 1 VGPR (Value: Float) (op 224): v11

 // Deleting tag 357
 // (op 698) Deallocate 359
 // Freeing : 1 VGPR (Value: Float) (op 228): v10

 // Deleting tag 359
 // (op 41) Deallocate{}(698) END
 // (op 41) SetCoordinate(231) BEGIN
 // (op 231) SetCoordinate (231): Coordinate 282 = 20:U32
 // (op 231) generate({})
 // (op 231) end: generate({})
 // (op 231) generate({233})
 // (op 231) LoadLDSTile{Value: Float}(233) BEGIN
 // (op 233) GEN: loadMacroTileWAVELDS OP 233 LDS 14 WaveTile 87
 // (op 233) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 233) Generate 256:U32 into nullptr
 // tag 361: v**UNALLOCATED**
 // (op 233) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 233) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 10240
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 233): v11
ds_read_b32 v11, v0 offset:10240 // (op 233) Load local data 
// VMEM: Expected complete at 991 (current 971)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 231) LoadLDSTile{Value: Float}(233) END
 // (op 231) end: generate({233})
 // (op 41) SetCoordinate(231) END
 // (op 41) SetCoordinate(235) BEGIN
 // (op 235) SetCoordinate (235): Coordinate 282 = 20:U32
 // (op 235) generate({})
 // (op 235) end: generate({})
 // (op 235) generate({237})
 // (op 235) LoadLDSTile{Value: Float}(237) BEGIN
 // (op 237) GEN: loadMacroTileWAVELDS OP 237 LDS 22 WaveTile 147
 // (op 237) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 237) Generate 256:U32 into nullptr
 // tag 363: v**UNALLOCATED**
 // (op 237) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 237) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 26624
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 237): v10
ds_read_b32 v10, v4 offset:26624 // (op 237) Load local data 
// VMEM: Expected complete at 992 (current 972)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 235) LoadLDSTile{Value: Float}(237) END
 // (op 235) end: generate({237})
 // (op 41) SetCoordinate(235) END
 // (op 41) Multiply(239) BEGIN
 // (op 239) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 239) 
 // (op 41) Multiply(239) END
 // (op 41) Deallocate{}(700) BEGIN
 // (op 700) Deallocate 361
 // Freeing : 1 VGPR (Value: Float) (op 233): v11

 // Deleting tag 361
 // (op 700) Deallocate 363
 // Freeing : 1 VGPR (Value: Float) (op 237): v10

 // Deleting tag 363
 // (op 41) Deallocate{}(700) END
 // (op 41) SetCoordinate(240) BEGIN
 // (op 240) SetCoordinate (240): Coordinate 282 = 21:U32
 // (op 240) generate({})
 // (op 240) end: generate({})
 // (op 240) generate({242})
 // (op 240) LoadLDSTile{Value: Float}(242) BEGIN
 // (op 242) GEN: loadMacroTileWAVELDS OP 242 LDS 14 WaveTile 87
 // (op 242) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 242) Generate 256:U32 into nullptr
 // tag 365: v**UNALLOCATED**
 // (op 242) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 242) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 10752
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 242): v11
ds_read_b32 v11, v0 offset:10752 // (op 242) Load local data 
// VMEM: Expected complete at 1029 (current 1009)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 240) LoadLDSTile{Value: Float}(242) END
 // (op 240) end: generate({242})
 // (op 41) SetCoordinate(240) END
 // (op 41) SetCoordinate(244) BEGIN
 // (op 244) SetCoordinate (244): Coordinate 282 = 21:U32
 // (op 244) generate({})
 // (op 244) end: generate({})
 // (op 244) generate({246})
 // (op 244) LoadLDSTile{Value: Float}(246) BEGIN
 // (op 246) GEN: loadMacroTileWAVELDS OP 246 LDS 22 WaveTile 147
 // (op 246) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 246) Generate 256:U32 into nullptr
 // tag 367: v**UNALLOCATED**
 // (op 246) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 246) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 27136
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 246): v10
ds_read_b32 v10, v4 offset:27136 // (op 246) Load local data 
// VMEM: Expected complete at 1030 (current 1010)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 244) LoadLDSTile{Value: Float}(246) END
 // (op 244) end: generate({246})
 // (op 41) SetCoordinate(244) END
 // (op 41) Multiply(248) BEGIN
 // (op 248) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 248) 
 // (op 41) Multiply(248) END
 // (op 41) Deallocate{}(702) BEGIN
 // (op 702) Deallocate 365
 // Freeing : 1 VGPR (Value: Float) (op 242): v11

 // Deleting tag 365
 // (op 702) Deallocate 367
 // Freeing : 1 VGPR (Value: Float) (op 246): v10

 // Deleting tag 367
 // (op 41) Deallocate{}(702) END
 // (op 41) SetCoordinate(249) BEGIN
 // (op 249) SetCoordinate (249): Coordinate 282 = 22:U32
 // (op 249) generate({})
 // (op 249) end: generate({})
 // (op 249) generate({251})
 // (op 249) LoadLDSTile{Value: Float}(251) BEGIN
 // (op 251) GEN: loadMacroTileWAVELDS OP 251 LDS 14 WaveTile 87
 // (op 251) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 251) Generate 256:U32 into nullptr
 // tag 369: v**UNALLOCATED**
 // (op 251) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 251) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 11264
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 251): v11
ds_read_b32 v11, v0 offset:11264 // (op 251) Load local data 
// VMEM: Expected complete at 1067 (current 1047)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 249) LoadLDSTile{Value: Float}(251) END
 // (op 249) end: generate({251})
 // (op 41) SetCoordinate(249) END
 // (op 41) SetCoordinate(253) BEGIN
 // (op 253) SetCoordinate (253): Coordinate 282 = 22:U32
 // (op 253) generate({})
 // (op 253) end: generate({})
 // (op 253) generate({255})
 // (op 253) LoadLDSTile{Value: Float}(255) BEGIN
 // (op 255) GEN: loadMacroTileWAVELDS OP 255 LDS 22 WaveTile 147
 // (op 255) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 255) Generate 256:U32 into nullptr
 // tag 371: v**UNALLOCATED**
 // (op 255) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 255) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 27648
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 255): v10
ds_read_b32 v10, v4 offset:27648 // (op 255) Load local data 
// VMEM: Expected complete at 1068 (current 1048)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 253) LoadLDSTile{Value: Float}(255) END
 // (op 253) end: generate({255})
 // (op 41) SetCoordinate(253) END
 // (op 41) Multiply(257) BEGIN
 // (op 257) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 257) 
 // (op 41) Multiply(257) END
 // (op 41) Deallocate{}(704) BEGIN
 // (op 704) Deallocate 369
 // Freeing : 1 VGPR (Value: Float) (op 251): v11

 // Deleting tag 369
 // (op 704) Deallocate 371
 // Freeing : 1 VGPR (Value: Float) (op 255): v10

 // Deleting tag 371
 // (op 41) Deallocate{}(704) END
 // (op 41) SetCoordinate(258) BEGIN
 // (op 258) SetCoordinate (258): Coordinate 282 = 23:U32
 // (op 258) generate({})
 // (op 258) end: generate({})
 // (op 258) generate({260})
 // (op 258) LoadLDSTile{Value: Float}(260) BEGIN
 // (op 260) GEN: loadMacroTileWAVELDS OP 260 LDS 14 WaveTile 87
 // (op 260) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 260) Generate 256:U32 into nullptr
 // tag 373: v**UNALLOCATED**
 // (op 260) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 260) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 11776
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 260): v11
ds_read_b32 v11, v0 offset:11776 // (op 260) Load local data 
// VMEM: Expected complete at 1105 (current 1085)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 258) LoadLDSTile{Value: Float}(260) END
 // (op 258) end: generate({260})
 // (op 41) SetCoordinate(258) END
 // (op 41) SetCoordinate(262) BEGIN
 // (op 262) SetCoordinate (262): Coordinate 282 = 23:U32
 // (op 262) generate({})
 // (op 262) end: generate({})
 // (op 262) generate({264})
 // (op 262) LoadLDSTile{Value: Float}(264) BEGIN
 // (op 264) GEN: loadMacroTileWAVELDS OP 264 LDS 22 WaveTile 147
 // (op 264) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 264) Generate 256:U32 into nullptr
 // tag 375: v**UNALLOCATED**
 // (op 264) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 264) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 28160
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 264): v10
ds_read_b32 v10, v4 offset:28160 // (op 264) Load local data 
// VMEM: Expected complete at 1106 (current 1086)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 262) LoadLDSTile{Value: Float}(264) END
 // (op 262) end: generate({264})
 // (op 41) SetCoordinate(262) END
 // (op 41) Multiply(266) BEGIN
 // (op 266) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 266) 
 // (op 41) Multiply(266) END
 // (op 41) Deallocate{}(706) BEGIN
 // (op 706) Deallocate 373
 // Freeing : 1 VGPR (Value: Float) (op 260): v11

 // Deleting tag 373
 // (op 706) Deallocate 375
 // Freeing : 1 VGPR (Value: Float) (op 264): v10

 // Deleting tag 375
 // (op 41) Deallocate{}(706) END
 // (op 41) SetCoordinate(267) BEGIN
 // (op 267) SetCoordinate (267): Coordinate 282 = 24:U32
 // (op 267) generate({})
 // (op 267) end: generate({})
 // (op 267) generate({269})
 // (op 267) LoadLDSTile{Value: Float}(269) BEGIN
 // (op 269) GEN: loadMacroTileWAVELDS OP 269 LDS 14 WaveTile 87
 // (op 269) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 269) Generate 256:U32 into nullptr
 // tag 377: v**UNALLOCATED**
 // (op 269) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 269) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 12288
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 269): v11
ds_read_b32 v11, v0 offset:12288 // (op 269) Load local data 
// VMEM: Expected complete at 1143 (current 1123)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 267) LoadLDSTile{Value: Float}(269) END
 // (op 267) end: generate({269})
 // (op 41) SetCoordinate(267) END
 // (op 41) SetCoordinate(271) BEGIN
 // (op 271) SetCoordinate (271): Coordinate 282 = 24:U32
 // (op 271) generate({})
 // (op 271) end: generate({})
 // (op 271) generate({273})
 // (op 271) LoadLDSTile{Value: Float}(273) BEGIN
 // (op 273) GEN: loadMacroTileWAVELDS OP 273 LDS 22 WaveTile 147
 // (op 273) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 273) Generate 256:U32 into nullptr
 // tag 379: v**UNALLOCATED**
 // (op 273) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 273) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 28672
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 273): v10
ds_read_b32 v10, v4 offset:28672 // (op 273) Load local data 
// VMEM: Expected complete at 1144 (current 1124)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 271) LoadLDSTile{Value: Float}(273) END
 // (op 271) end: generate({273})
 // (op 41) SetCoordinate(271) END
 // (op 41) Multiply(275) BEGIN
 // (op 275) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 275) 
 // (op 41) Multiply(275) END
 // (op 41) Deallocate{}(708) BEGIN
 // (op 708) Deallocate 379
 // Freeing : 1 VGPR (Value: Float) (op 273): v10

 // Deleting tag 379
 // (op 708) Deallocate 377
 // Freeing : 1 VGPR (Value: Float) (op 269): v11

 // Deleting tag 377
 // (op 41) Deallocate{}(708) END
 // (op 41) SetCoordinate(276) BEGIN
 // (op 276) SetCoordinate (276): Coordinate 282 = 25:U32
 // (op 276) generate({})
 // (op 276) end: generate({})
 // (op 276) generate({278})
 // (op 276) LoadLDSTile{Value: Float}(278) BEGIN
 // (op 278) GEN: loadMacroTileWAVELDS OP 278 LDS 14 WaveTile 87
 // (op 278) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 278) Generate 256:U32 into nullptr
 // tag 381: v**UNALLOCATED**
 // (op 278) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 278) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 12800
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 278): v11
ds_read_b32 v11, v0 offset:12800 // (op 278) Load local data 
// VMEM: Expected complete at 1181 (current 1161)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 276) LoadLDSTile{Value: Float}(278) END
 // (op 276) end: generate({278})
 // (op 41) SetCoordinate(276) END
 // (op 41) SetCoordinate(280) BEGIN
 // (op 280) SetCoordinate (280): Coordinate 282 = 25:U32
 // (op 280) generate({})
 // (op 280) end: generate({})
 // (op 280) generate({282})
 // (op 280) LoadLDSTile{Value: Float}(282) BEGIN
 // (op 282) GEN: loadMacroTileWAVELDS OP 282 LDS 22 WaveTile 147
 // (op 282) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 282) Generate 256:U32 into nullptr
 // tag 383: v**UNALLOCATED**
 // (op 282) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 282) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 29184
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 282): v10
ds_read_b32 v10, v4 offset:29184 // (op 282) Load local data 
// VMEM: Expected complete at 1182 (current 1162)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 280) LoadLDSTile{Value: Float}(282) END
 // (op 280) end: generate({282})
 // (op 41) SetCoordinate(280) END
 // (op 41) Multiply(284) BEGIN
 // (op 284) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 284) 
 // (op 41) Multiply(284) END
 // (op 41) Deallocate{}(710) BEGIN
 // (op 710) Deallocate 383
 // Freeing : 1 VGPR (Value: Float) (op 282): v10

 // Deleting tag 383
 // (op 710) Deallocate 381
 // Freeing : 1 VGPR (Value: Float) (op 278): v11

 // Deleting tag 381
 // (op 41) Deallocate{}(710) END
 // (op 41) SetCoordinate(285) BEGIN
 // (op 285) SetCoordinate (285): Coordinate 282 = 26:U32
 // (op 285) generate({})
 // (op 285) end: generate({})
 // (op 285) generate({287})
 // (op 285) LoadLDSTile{Value: Float}(287) BEGIN
 // (op 287) GEN: loadMacroTileWAVELDS OP 287 LDS 14 WaveTile 87
 // (op 287) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 287) Generate 256:U32 into nullptr
 // tag 385: v**UNALLOCATED**
 // (op 287) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 287) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 13312
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 287): v11
ds_read_b32 v11, v0 offset:13312 // (op 287) Load local data 
// VMEM: Expected complete at 1219 (current 1199)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 285) LoadLDSTile{Value: Float}(287) END
 // (op 285) end: generate({287})
 // (op 41) SetCoordinate(285) END
 // (op 41) SetCoordinate(289) BEGIN
 // (op 289) SetCoordinate (289): Coordinate 282 = 26:U32
 // (op 289) generate({})
 // (op 289) end: generate({})
 // (op 289) generate({291})
 // (op 289) LoadLDSTile{Value: Float}(291) BEGIN
 // (op 291) GEN: loadMacroTileWAVELDS OP 291 LDS 22 WaveTile 147
 // (op 291) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 291) Generate 256:U32 into nullptr
 // tag 387: v**UNALLOCATED**
 // (op 291) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 291) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 29696
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 291): v10
ds_read_b32 v10, v4 offset:29696 // (op 291) Load local data 
// VMEM: Expected complete at 1220 (current 1200)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 289) LoadLDSTile{Value: Float}(291) END
 // (op 289) end: generate({291})
 // (op 41) SetCoordinate(289) END
 // (op 41) Multiply(293) BEGIN
 // (op 293) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 293) 
 // (op 41) Multiply(293) END
 // (op 41) Deallocate{}(712) BEGIN
 // (op 712) Deallocate 387
 // Freeing : 1 VGPR (Value: Float) (op 291): v10

 // Deleting tag 387
 // (op 712) Deallocate 385
 // Freeing : 1 VGPR (Value: Float) (op 287): v11

 // Deleting tag 385
 // (op 41) Deallocate{}(712) END
 // (op 41) SetCoordinate(294) BEGIN
 // (op 294) SetCoordinate (294): Coordinate 282 = 27:U32
 // (op 294) generate({})
 // (op 294) end: generate({})
 // (op 294) generate({296})
 // (op 294) LoadLDSTile{Value: Float}(296) BEGIN
 // (op 296) GEN: loadMacroTileWAVELDS OP 296 LDS 14 WaveTile 87
 // (op 296) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 296) Generate 256:U32 into nullptr
 // tag 389: v**UNALLOCATED**
 // (op 296) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 296) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 13824
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 296): v11
ds_read_b32 v11, v0 offset:13824 // (op 296) Load local data 
// VMEM: Expected complete at 1257 (current 1237)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 294) LoadLDSTile{Value: Float}(296) END
 // (op 294) end: generate({296})
 // (op 41) SetCoordinate(294) END
 // (op 41) SetCoordinate(298) BEGIN
 // (op 298) SetCoordinate (298): Coordinate 282 = 27:U32
 // (op 298) generate({})
 // (op 298) end: generate({})
 // (op 298) generate({300})
 // (op 298) LoadLDSTile{Value: Float}(300) BEGIN
 // (op 300) GEN: loadMacroTileWAVELDS OP 300 LDS 22 WaveTile 147
 // (op 300) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 300) Generate 256:U32 into nullptr
 // tag 391: v**UNALLOCATED**
 // (op 300) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 300) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 30208
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 300): v10
ds_read_b32 v10, v4 offset:30208 // (op 300) Load local data 
// VMEM: Expected complete at 1258 (current 1238)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 298) LoadLDSTile{Value: Float}(300) END
 // (op 298) end: generate({300})
 // (op 41) SetCoordinate(298) END
 // (op 41) Multiply(302) BEGIN
 // (op 302) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 302) 
 // (op 41) Multiply(302) END
 // (op 41) Deallocate{}(714) BEGIN
 // (op 714) Deallocate 391
 // Freeing : 1 VGPR (Value: Float) (op 300): v10

 // Deleting tag 391
 // (op 714) Deallocate 389
 // Freeing : 1 VGPR (Value: Float) (op 296): v11

 // Deleting tag 389
 // (op 41) Deallocate{}(714) END
 // (op 41) SetCoordinate(303) BEGIN
 // (op 303) SetCoordinate (303): Coordinate 282 = 28:U32
 // (op 303) generate({})
 // (op 303) end: generate({})
 // (op 303) generate({305})
 // (op 303) LoadLDSTile{Value: Float}(305) BEGIN
 // (op 305) GEN: loadMacroTileWAVELDS OP 305 LDS 14 WaveTile 87
 // (op 305) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 305) Generate 256:U32 into nullptr
 // tag 393: v**UNALLOCATED**
 // (op 305) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 305) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 14336
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 305): v11
ds_read_b32 v11, v0 offset:14336 // (op 305) Load local data 
// VMEM: Expected complete at 1295 (current 1275)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 303) LoadLDSTile{Value: Float}(305) END
 // (op 303) end: generate({305})
 // (op 41) SetCoordinate(303) END
 // (op 41) SetCoordinate(307) BEGIN
 // (op 307) SetCoordinate (307): Coordinate 282 = 28:U32
 // (op 307) generate({})
 // (op 307) end: generate({})
 // (op 307) generate({309})
 // (op 307) LoadLDSTile{Value: Float}(309) BEGIN
 // (op 309) GEN: loadMacroTileWAVELDS OP 309 LDS 22 WaveTile 147
 // (op 309) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 309) Generate 256:U32 into nullptr
 // tag 395: v**UNALLOCATED**
 // (op 309) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 309) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 30720
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 309): v10
ds_read_b32 v10, v4 offset:30720 // (op 309) Load local data 
// VMEM: Expected complete at 1296 (current 1276)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 307) LoadLDSTile{Value: Float}(309) END
 // (op 307) end: generate({309})
 // (op 41) SetCoordinate(307) END
 // (op 41) Multiply(311) BEGIN
 // (op 311) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 311) 
 // (op 41) Multiply(311) END
 // (op 41) Deallocate{}(716) BEGIN
 // (op 716) Deallocate 393
 // Freeing : 1 VGPR (Value: Float) (op 305): v11

 // Deleting tag 393
 // (op 716) Deallocate 395
 // Freeing : 1 VGPR (Value: Float) (op 309): v10

 // Deleting tag 395
 // (op 41) Deallocate{}(716) END
 // (op 41) SetCoordinate(312) BEGIN
 // (op 312) SetCoordinate (312): Coordinate 282 = 29:U32
 // (op 312) generate({})
 // (op 312) end: generate({})
 // (op 312) generate({314})
 // (op 312) LoadLDSTile{Value: Float}(314) BEGIN
 // (op 314) GEN: loadMacroTileWAVELDS OP 314 LDS 14 WaveTile 87
 // (op 314) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 314) Generate 256:U32 into nullptr
 // tag 397: v**UNALLOCATED**
 // (op 314) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 314) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 14848
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 314): v11
ds_read_b32 v11, v0 offset:14848 // (op 314) Load local data 
// VMEM: Expected complete at 1333 (current 1313)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 312) LoadLDSTile{Value: Float}(314) END
 // (op 312) end: generate({314})
 // (op 41) SetCoordinate(312) END
 // (op 41) SetCoordinate(316) BEGIN
 // (op 316) SetCoordinate (316): Coordinate 282 = 29:U32
 // (op 316) generate({})
 // (op 316) end: generate({})
 // (op 316) generate({318})
 // (op 316) LoadLDSTile{Value: Float}(318) BEGIN
 // (op 318) GEN: loadMacroTileWAVELDS OP 318 LDS 22 WaveTile 147
 // (op 318) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 318) Generate 256:U32 into nullptr
 // tag 399: v**UNALLOCATED**
 // (op 318) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 318) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 31232
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 318): v10
ds_read_b32 v10, v4 offset:31232 // (op 318) Load local data 
// VMEM: Expected complete at 1334 (current 1314)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 316) LoadLDSTile{Value: Float}(318) END
 // (op 316) end: generate({318})
 // (op 41) SetCoordinate(316) END
 // (op 41) Multiply(320) BEGIN
 // (op 320) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 320) 
 // (op 41) Multiply(320) END
 // (op 41) Deallocate{}(718) BEGIN
 // (op 718) Deallocate 397
 // Freeing : 1 VGPR (Value: Float) (op 314): v11

 // Deleting tag 397
 // (op 718) Deallocate 399
 // Freeing : 1 VGPR (Value: Float) (op 318): v10

 // Deleting tag 399
 // (op 41) Deallocate{}(718) END
 // (op 41) SetCoordinate(321) BEGIN
 // (op 321) SetCoordinate (321): Coordinate 282 = 30:U32
 // (op 321) generate({})
 // (op 321) end: generate({})
 // (op 321) generate({323})
 // (op 321) LoadLDSTile{Value: Float}(323) BEGIN
 // (op 323) GEN: loadMacroTileWAVELDS OP 323 LDS 14 WaveTile 87
 // (op 323) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 323) Generate 256:U32 into nullptr
 // tag 401: v**UNALLOCATED**
 // (op 323) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 323) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 15360
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 323): v11
ds_read_b32 v11, v0 offset:15360 // (op 323) Load local data 
// VMEM: Expected complete at 1371 (current 1351)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 321) LoadLDSTile{Value: Float}(323) END
 // (op 321) end: generate({323})
 // (op 41) SetCoordinate(321) END
 // (op 41) SetCoordinate(325) BEGIN
 // (op 325) SetCoordinate (325): Coordinate 282 = 30:U32
 // (op 325) generate({})
 // (op 325) end: generate({})
 // (op 325) generate({327})
 // (op 325) LoadLDSTile{Value: Float}(327) BEGIN
 // (op 327) GEN: loadMacroTileWAVELDS OP 327 LDS 22 WaveTile 147
 // (op 327) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 327) Generate 256:U32 into nullptr
 // tag 403: v**UNALLOCATED**
 // (op 327) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 327) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 31744
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 327): v10
ds_read_b32 v10, v4 offset:31744 // (op 327) Load local data 
// VMEM: Expected complete at 1372 (current 1352)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 325) LoadLDSTile{Value: Float}(327) END
 // (op 325) end: generate({327})
 // (op 41) SetCoordinate(325) END
 // (op 41) Multiply(329) BEGIN
 // (op 329) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 329) 
 // (op 41) Multiply(329) END
 // (op 41) Deallocate{}(720) BEGIN
 // (op 720) Deallocate 401
 // Freeing : 1 VGPR (Value: Float) (op 323): v11

 // Deleting tag 401
 // (op 720) Deallocate 403
 // Freeing : 1 VGPR (Value: Float) (op 327): v10

 // Deleting tag 403
 // (op 41) Deallocate{}(720) END
 // (op 41) SetCoordinate(330) BEGIN
 // (op 330) SetCoordinate (330): Coordinate 282 = 31:U32
 // (op 330) generate({})
 // (op 330) end: generate({})
 // (op 330) generate({332})
 // (op 330) LoadLDSTile{Value: Float}(332) BEGIN
 // (op 332) GEN: loadMacroTileWAVELDS OP 332 LDS 14 WaveTile 87
 // (op 332) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 332) Generate 256:U32 into nullptr
 // tag 405: v**UNALLOCATED**
 // (op 332) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // 	colStrideIsOne = 0

 // (op 332) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset593: VGPR Value: UInt32 x 1: v0
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 15872
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 332): v11
ds_read_b32 v11, v0 offset:15872 // (op 332) Load local data 
// VMEM: Expected complete at 1409 (current 1389)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 330) LoadLDSTile{Value: Float}(332) END
 // (op 330) end: generate({332})
 // (op 41) SetCoordinate(330) END
 // (op 41) SetCoordinate(334) BEGIN
 // (op 334) SetCoordinate (334): Coordinate 282 = 31:U32
 // (op 334) generate({})
 // (op 334) end: generate({})
 // (op 334) generate({336})
 // (op 334) LoadLDSTile{Value: Float}(336) BEGIN
 // (op 336) GEN: loadMacroTileWAVELDS OP 336 LDS 22 WaveTile 147
 // (op 336) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 336) Generate 256:U32 into nullptr
 // tag 407: v**UNALLOCATED**
 // (op 336) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 16384
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 0

 // (op 336) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 1
 // 	info.n = 1
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 1: (unallocated)
 // 	info.rowOffsetReg = Offset602: VGPR Value: UInt32 x 1: v4
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 0
 // 	info.rowStrideAttributes = {
 // 	t.dataType = None
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = nullptr
 // 	t.trLoadPairStride = nullptr
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 32256
 // 	info.bufDesc = (nullptr)
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
// Allocated : 1 VGPR (Value: Float) (op 336): v10
ds_read_b32 v10, v4 offset:32256 // (op 336) Load local data 
// VMEM: Expected complete at 1410 (current 1390)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 16384, Size: 16384)
 // (op 334) LoadLDSTile{Value: Float}(336) END
 // (op 334) end: generate({336})
 // (op 41) SetCoordinate(334) END
 // (op 41) Multiply(338) BEGIN
 // (op 338) Generate MatrixMultiply(WaveTileS, WaveTileS, {DataFlowTag36: a[0:15]:S})S into DataFlowTag36: ACCVGPR Value: Float x 16: a[0:15]
s_waitcnt lgkmcnt(0)
v_mfma_f32_32x32x2f32 a[0:15], v11, v10, a[0:15] // (op 338) 
 // (op 41) Multiply(338) END
 // (op 41) Deallocate{}(722) BEGIN
 // (op 722) Deallocate 405
 // Freeing : 1 VGPR (Value: Float) (op 332): v11

 // Deleting tag 405
 // (op 722) Deallocate 407
 // Freeing : 1 VGPR (Value: Float) (op 336): v10

 // Deleting tag 407
 // (op 41) Deallocate{}(722) END
 // (op 41) end: generate({4, 43})
 // (op 41) For Loop Increment
 // (op 41) generate({})
 // (op 41) end: generate({})
 // (op 41) Condition: Bottom (jump to GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ForLoopTop_KLoop_41 if true)
 // (op 41) Generate LessThan(DataFlowTag(277)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_9:I64, 6:U32)I64)U32)BL into SCC Value: Bool x 1: scc
 // (op 41) Get arg Tensor_0_size_1_9
 // (op 41) LessThan({DataFlowTag277: s1:U32}, Convert(ArithmeticShiftR({Tensor_0_size_1_9: s[20:21]:I64}, 6:U32)I64)U32)BL
 // (op 41) ArithmeticShiftR({Tensor_0_size_1_9: s[20:21]:I64}, 6:U32)I64
// Allocated : 2 SGPRs (Value: Int64) (op 41): s6, s7
s_ashr_i64 s[6:7], s[20:21], 6 // (op 41) 
// Allocated : 1 SGPR (Value: UInt32) (op 41): s0
s_mov_b32 s0, s6 // (op 41) convert
 // Freeing : 2 SGPRs (Value: Int64) (op 41): s6, s7

s_cmp_lt_u32 s1, s0 // (op 41) 
 // Freeing : 1 SGPR (Value: UInt32) (op 41): s0

s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 41) 
s_cbranch_scc1 GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ForLoopTop_KLoop_41 // (op 41) Condition: Bottom (jump to GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ForLoopTop_KLoop_41 if true)
GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ForLoopBottom_KLoop_41:
 // (op 41) 
s_waitcnt vmcnt(0) lgkmcnt(0) expcnt(0)// DEBUG: Wait after branch
 // (op 41) 
 // (op 41) Unlock For Loop
 // (op 514) ForLoopOp KLoop: LessThan(DataFlowTag(277)U32, Convert(ArithmeticShiftR(Tensor_0_size_1_9:I64, 6:U32)I64)U32)BL(41) END
 // (op 514) Deallocate{Tensor_0_size_1_9, Tensor_2_extent_2, Tensor_2_stride_0_11, Tensor_0_extent_0, Tensor_0_stride_1_10}(658) BEGIN
 // (op 658) Deallocate 14
 // Deleting tag 14
 // (op 658) Deallocate 22
 // Deleting tag 22
 // (op 658) Deallocate 277
 // Freeing DataFlowTag277: 1 SGPR (Value: UInt32) (op 42): s1

 // Deleting tag 277
 // (op 658) Deallocate Tensor_0_size_1_9
 // Freeing Tensor_0_size_1_9: 2 SGPRs (Value: Raw32): s20, s21

 // (op 658) Deallocate Tensor_2_extent_2
 // Freeing Tensor_2_extent_2: 2 SGPRs (Value: Raw32): s8, s9

 // (op 658) Deallocate Tensor_2_stride_0_11
 // Freeing Tensor_2_stride_0_11: 2 SGPRs (Value: Raw32): s24, s25

 // (op 658) Deallocate Tensor_0_extent_0
 // Freeing Tensor_0_extent_0: 2 SGPRs (Value: Raw32): s4, s5

 // (op 658) Deallocate Tensor_0_stride_1_10
 // Freeing Tensor_0_stride_1_10: 2 SGPRs (Value: Raw32): s22, s23

 // (op 514) Deallocate{Tensor_0_size_1_9, Tensor_2_extent_2, Tensor_2_stride_0_11, Tensor_0_extent_0, Tensor_0_stride_1_10}(658) END
 // (op 514) end: generate({519})
 // Freeing offset455: 1 VGPR (Value: UInt32) (op 10): v13

 // Deleting tag 455
 // Freeing offset447: 1 VGPR (Value: UInt32) (op 4): v12

 // Deleting tag 447
 // Freeing Offset588: 1 VGPR (Value: UInt32) (op 588): v7

 // Deleting tag 465
 // Freeing Offset560: 1 VGPR (Value: UInt32) (op 560): v8

 // Deleting tag 443
 // Freeing Offset597: 1 VGPR (Value: UInt32) (op 597): v6

 // Deleting tag 473
 // Freeing Buffer446: 4 SGPRs (Buffer: None) (op 560): s36, s37, s38, s39

 // Deleting tag 446
 // Freeing Offset571: 1 VGPR (Value: UInt32) (op 571): v9

 // Deleting tag 451
 // Freeing Buffer454: 4 SGPRs (Buffer: None) (op 571): s40, s41, s42, s43

 // Deleting tag 454
 // (op 514) Unlock Scope 514
 // (op 1) Scope(514) END
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(25)S)BL(467) BEGIN
 // (op 467) Lock for Conditional
 // (op 467) Generate Equal(0.00000:S, DataFlowTag(25)S)BL into VCC Value: Bool64 x 1: vcc
 // (op 467) reg expression
 // (op 467) Equal(0.00000:S, v3:S)BL64
v_cmp_eq_f32 vcc, 0.00000, v3 // (op 467) 
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 467) 
s_cbranch_vccz GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ConditionalFalse_0_beta_467 // (op 467) Condition: False, jump to GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ConditionalFalse_0_beta_467
 // (op 467) generate({522})
 // (op 467) Scope(522) BEGIN
 // (op 522) Lock Scope 522
 // (op 522) generate({525})
 // (op 522) NOP(525) BEGIN
 // (op 525) generate({})
 // (op 525) end: generate({})
 // (op 522) NOP(525) END
 // (op 522) Assign VGPR Multiply(DataFlowTag(34)NA, DataFlowTag(36)NA)NA(29) BEGIN
 // (op 29) Assign dim(40) = Multiply(DataFlowTag(34)NA, DataFlowTag(36)NA)NA
 // (op 29) Generate Multiply(DataFlowTag(34)NA, DataFlowTag(36)NA)NA into nullptr
 // (op 29) reg expression
 // (op 29) reg expression
 // (op 29) Multiply(v5:S, {DataFlowTag36: a[0:15]:S})S
// Allocated : 16 VGPRs (Value: Float) (op 29): v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21
s_nop 12
v_accvgpr_read v6, a0 // (op 29) Wait state hazard: XDL Write Hazard
v_accvgpr_read v7, a1 // (op 29) 
v_accvgpr_read v8, a2 // (op 29) 
v_accvgpr_read v9, a3 // (op 29) 
v_accvgpr_read v10, a4 // (op 29) 
v_accvgpr_read v11, a5 // (op 29) 
v_accvgpr_read v12, a6 // (op 29) 
v_accvgpr_read v13, a7 // (op 29) 
v_accvgpr_read v14, a8 // (op 29) 
v_accvgpr_read v15, a9 // (op 29) 
v_accvgpr_read v16, a10 // (op 29) 
v_accvgpr_read v17, a11 // (op 29) 
v_accvgpr_read v18, a12 // (op 29) 
v_accvgpr_read v19, a13 // (op 29) 
v_accvgpr_read v20, a14 // (op 29) 
v_accvgpr_read v21, a15 // (op 29) 
// Allocated : 16 VGPRs (Value: Float) (op 29): v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37
v_mul_f32 v22, v5, v6 // (op 29) 
v_mul_f32 v23, v5, v7 // (op 29) 
v_mul_f32 v24, v5, v8 // (op 29) 
v_mul_f32 v25, v5, v9 // (op 29) 
v_mul_f32 v26, v5, v10 // (op 29) 
v_mul_f32 v27, v5, v11 // (op 29) 
v_mul_f32 v28, v5, v12 // (op 29) 
v_mul_f32 v29, v5, v13 // (op 29) 
v_mul_f32 v30, v5, v14 // (op 29) 
v_mul_f32 v31, v5, v15 // (op 29) 
v_mul_f32 v32, v5, v16 // (op 29) 
v_mul_f32 v33, v5, v17 // (op 29) 
v_mul_f32 v34, v5, v18 // (op 29) 
v_mul_f32 v35, v5, v19 // (op 29) 
v_mul_f32 v36, v5, v20 // (op 29) 
v_mul_f32 v37, v5, v21 // (op 29) 
 // Freeing : 16 VGPRs (Value: Float) (op 29): v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21

 // (op 522) Assign VGPR Multiply(DataFlowTag(34)NA, DataFlowTag(36)NA)NA(29) END
 // (op 522) Barrier(548) BEGIN
s_barrier  // (op 548) 
 // (op 522) Barrier(548) END
 // (op 522) Scope(609) BEGIN
 // (op 609) Lock Scope 609
 // (op 609) generate({606})
 // (op 609) ComputeIndex(606) BEGIN
 // (op 606) KernelGraph::LoadStoreTileGenerator::ComputeIndex(606): target 42 increment 218 base -1 offset 481 stride 482 buffer -1
 // tag 481: v**UNALLOCATED**
 // FastArithmetic:	orig = {Flatten: Add(Multiply({Flatten: Add(Multiply({Tile[1]: Modulo({Tile[0]: Divide({Workitem Index X: v1:U32}, 64:U32)U32}, 2:U32)U32}, 32:U32)U32, {Flatten: Add(Multiply({Tile[1]: Modulo({Flatten: Add(Multiply(0:U32, 16:U32)U32, {Tile[0]: Divide({Tile[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 4:U32)U32})U32}, 8:U32)U32}, 4:U32)U32, {Tile[1]: Modulo({Tile[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 4:U32)U32})U32})U32}, 64:U32)U32, {Flatten: Add(Multiply({Tile[0]: Divide({Tile[0]: Divide({Workitem Index X: v1:U32}, 64:U32)U32}, 2:U32)U32}, 32:U32)U32, {Flatten: Add(Multiply({Tile[0]: Divide({Flatten: Add(Multiply(0:U32, 16:U32)U32, {Tile[0]: Divide({Tile[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 4:U32)U32})U32}, 8:U32)U32}, 4:I)U32, 0:U32)U32})U32})U32}
 // 	x = {Flatten: ShiftLAdd({Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32}, 6:U32, {Flatten: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}

 // (op 606)   Offset(481): indexExpr: {Flatten: ShiftLAdd({Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32}, 6:U32, {Flatten: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}
 // (op 606)   Offset(481): paddingBytes: 0:U32
 // (op 606) Generate Convert(Add(Multiply({Flatten: ShiftLAdd({Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32}, 6:U32, {Flatten: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, 4:U32)U32, 0:U32)U32)U32 into Offset606: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Flatten: ShiftLAdd({Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32}, 6:U32, {Flatten: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, 4:U32)U32, 0:U32)U32)U32
 // 	x = ShiftL({Flatten: ShiftLAdd({Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32}, 6:U32, {Flatten: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, 2:U32)U32

 // (op 606) reg expression
 // (op 606) BEGIN: Tile[1]
 // (op 606) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 606) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 606): v21
v_and_b32 v21, 63, v1 // (op 606) 
 // (op 606) END: Tile[1]
// Allocated : 1 VGPR (Value: UInt32) (op 606): v20
v_lshrrev_b32 v20, 6, v1 // (op 606) 
 // (op 606) LogicalShiftR({Tile[1]: v21:U32}, 2:U32)U32
 // (op 606) BEGIN: Tile[1]
 // (op 606) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 606): v19
v_lshrrev_b32 v19, 2, v21 // (op 606) 
// Allocated : 1 VGPR (Value: UInt32) (op 606): v18
v_and_b32 v18, 3, v1 // (op 606) 
 // (op 606) END: Tile[1]
 // (op 606) BEGIN: Tile[1]
 // (op 606) {Tile[1]: BitwiseAnd(v19:U32, 7:U32)U32}
 // (op 606) LogicalShiftR({Tile[1]: v21:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 606): v17
v_and_b32 v17, 7, v19 // (op 606) 
 // (op 606) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 606): v19

// Allocated : 1 VGPR (Value: UInt32) (op 606): v19
v_lshrrev_b32 v19, 5, v21 // (op 606) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 606): v21

 // (op 606) BEGIN: Flatten
 // (op 606) BEGIN: Tile[1]
 // (op 606) {Tile[1]: BitwiseAnd(v20:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 606): v21
v_lshl_add_u32 v21, v17, 2, v18 // (op 606) 
 // (op 606) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 606): v18

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 606): v17

// Allocated : 1 VGPR (Value: UInt32) (op 606): v18
v_and_b32 v18, 1, v20 // (op 606) 
 // (op 606) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 606): v20

 // (op 606) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 606) BEGIN: Flatten
 // (op 606) {Flatten: ShiftL(v19:U32, 2:U32)U32}
 // (op 606) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 606)         value (VGPR Value: UInt32 x 1: v19) 
 // (op 606)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 606): v20
v_lshrrev_b32 v20, 7, v1 // (op 606) 
// Allocated : 1 VGPR (Value: UInt32) (op 606): v17
v_lshlrev_b32 v17, 2, v19 // (op 606) 
 // (op 606) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 606): v19

 // (op 606) BEGIN: Flatten
 // (op 606) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 606): v19
v_lshl_add_u32 v19, v18, 5, v21 // (op 606) 
 // (op 606) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 606): v21

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 606): v18

// Allocated : 1 VGPR (Value: UInt32) (op 606): v18
v_lshl_add_u32 v18, v20, 5, v17 // (op 606) 
 // (op 606) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 606): v17

 // Freeing : 1 VGPR (Value: UInt32) (op 606): v20

 // (op 606) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 606): v17
v_lshl_add_u32 v17, v19, 6, v18 // (op 606) 
 // (op 606) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 606): v18

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 606): v19

 // (op 606) ShiftL({Flatten: v17:U32}, 2:U32)U32
 // (op 606) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 606)         value (Flatten: VGPR Value: UInt32 x 1: v17) 
 // (op 606)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated Offset606: 1 VGPR (Value: UInt32) (op 606): v16
v_lshlrev_b32 v16, 2, v17 // (op 606) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 606): v17

 // FastArithmetic:	orig = Add(Multiply(0:U32, Multiply(1:I, 64:U32)U32)U32, Multiply(8:U32, 1:I)U32)U32
 // 	x = 8:U32

 // (op 606)   Stride(482): indexExpr: 8:U32
 // (op 606)   Stride(482): indexExprPaddingBytes: 0:U32
 // (op 606)   Stride(482): unitStride: false vgprBlockSize: 0
 // (op 606)   Stride(482): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 606)   Stride(482): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(8:U32, 4:U32)U32, 0:U32)U32
 // 	x = 32:U32

 // (op 609) ComputeIndex(606) END
 // (op 609) ComputeIndex(607) BEGIN
 // (op 607) KernelGraph::LoadStoreTileGenerator::ComputeIndex(607): target 42 increment 219 base 481 offset 483 stride 484 buffer -1
 // FastArithmetic:	orig = Add(Multiply(0:U32, Multiply(1:I, 64:U32)U32)U32, Multiply(1:U32, 1:I)U32)U32
 // 	x = 1:U32

 // (op 607)   Stride(484): indexExpr: 1:U32
 // (op 607)   Stride(484): indexExprPaddingBytes: 0:U32
 // (op 607)   Stride(484): unitStride: true vgprBlockSize: 0
 // (op 607)   Stride(484): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 607)   Stride(484): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(1:U32, 4:U32)U32, 0:U32)U32
 // 	x = 4:U32

 // (op 609) ComputeIndex(607) END
 // (op 609) StoreLDSTile Value: Float(35) BEGIN
 // (op 35) GEN: StoreLDSTile
 // (op 35) GEN: storeMacroTileWAVELDS OP 35 LDS 42 MacroTile 40 WaveTile 213
 // (op 35) 	waveTile = WaveTile{1024:I}
 // 	waveTileNumElements = 1024
 // 	activeLanesInWave = 64
 // 	packing = 1
 // 	activeLanesInWave * packing = 64
 // 	store.varType = Value: Float

 // (op 35) 	agpr->description() = DataFlowTag40: VGPR Value: Float x 16: v[22:37]
 // 	waveTile = WaveTile{1024:I}
 // 	waveTileNumElements = 1024
 // 	activeLanesInWave = 64
 // 	packing = 1
 // 	activeLanesInWave * packing = 64

 // (op 35) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 35) Generate 32:U32 into nullptr
 // (op 35) Generate 4:U32 into nullptr
 // (op 35) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag40: VGPR Value: Float x 16: v[22:37]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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

 // (op 35) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag40: VGPR Value: Float x 16: v[22:37]
 // 	info.rowOffsetReg = Offset606: VGPR Value: UInt32 x 1: v16
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // (op 35) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 32
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_write_b128 v16, v[22:25] // (op 35) Store local data 
// VMEM: Expected complete at 1516 (current 1496)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
ds_write_b128 v16, v[26:29] offset:32 // (op 35) Store local data 
// VMEM: Expected complete at 1517 (current 1497)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
ds_write_b128 v16, v[30:33] offset:64 // (op 35) Store local data 
// VMEM: Expected complete at 1518 (current 1498)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
ds_write_b128 v16, v[34:37] offset:96 // (op 35) Store local data 
// VMEM: Expected complete at 1519 (current 1499)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 609) StoreLDSTile Value: Float(35) END
 // (op 609) end: generate({606})
 // Freeing Offset606: 1 VGPR (Value: UInt32) (op 606): v16

 // Deleting tag 481
 // (op 609) Unlock Scope 609
 // (op 522) Scope(609) END
 // (op 522) Deallocate{}(732) BEGIN
 // (op 732) Deallocate 40
 // Freeing DataFlowTag40: 16 VGPRs (Value: Float) (op 29): v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37

 // Deleting tag 40
 // (op 522) Deallocate{}(732) END
 // (op 522) Barrier(549) BEGIN
s_waitcnt lgkmcnt(0)
s_barrier  // (op 549) 
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 522) Barrier(549) END
 // (op 522) Scope(617) BEGIN
 // (op 617) Lock Scope 617
 // (op 617) generate({614})
 // (op 617) ComputeIndex(614) BEGIN
 // (op 614) KernelGraph::LoadStoreTileGenerator::ComputeIndex(614): target 42 increment 250 base -1 offset 485 stride 486 buffer -1
 // tag 485: v**UNALLOCATED**
 // FastArithmetic:	orig = {Tile: Add(Multiply({Tile: Add(Multiply({Flatten[0]: Divide({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32}, 64:U32)U32, {Tile: Add(Multiply({Flatten[1]: Modulo({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32})U32}
 // 	x = {Tile: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}

 // (op 614)   Offset(485): indexExpr: {Tile: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}
 // (op 614)   Offset(485): paddingBytes: 0:U32
 // (op 614) Generate Convert(Add(Multiply({Tile: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}, 4:U32)U32, 0:U32)U32)U32 into Offset614: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Tile: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}, 4:U32)U32, 0:U32)U32)U32
 // 	x = ShiftL({Tile: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32)U32}, 2:U32)U32})U32}, 2:U32)U32

 // (op 614) reg expression
 // (op 614) BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32
 // (op 614) LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 614): v6
v_and_b32 v6, -16, v1 // (op 614) 
// Allocated : 1 VGPR (Value: UInt32) (op 614): v7
v_lshrrev_b32 v7, 4, v1 // (op 614) 
 // (op 614) BEGIN: Flatten[1]
 // (op 614) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v6:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 614): v8
v_sub_u32 v8, v1, v6 // (op 614) 
 // (op 614) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 614): v6

 // (op 614) BEGIN: Tile
 // (op 614) {Tile: ShiftL({Flatten[1]: v8:U32}, 2:U32)U32}
 // (op 614) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 614)         value (Flatten[1]: VGPR Value: UInt32 x 1: v8) 
 // (op 614)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 614): v6
v_lshlrev_b32 v6, 2, v8 // (op 614) 
 // (op 614) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 614): v8

 // (op 614) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 614): v8
v_lshl_add_u32 v8, v7, 8, v6 // (op 614) 
 // (op 614) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 614): v6

 // Freeing : 1 VGPR (Value: UInt32) (op 614): v7

 // (op 614) ShiftL({Tile: v8:U32}, 2:U32)U32
 // (op 614) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 614)         value (Tile: VGPR Value: UInt32 x 1: v8) 
 // (op 614)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated Offset614: 1 VGPR (Value: UInt32) (op 614): v7
v_lshlrev_b32 v7, 2, v8 // (op 614) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 614): v8

 // FastArithmetic:	orig = Add(Multiply(1:I, 64:U32)U32, 0:I)U32
 // 	x = 64:U32

 // (op 614)   Stride(486): indexExpr: 64:U32
 // (op 614)   Stride(486): indexExprPaddingBytes: 0:U32
 // (op 614)   Stride(486): unitStride: false vgprBlockSize: 0
 // (op 614)   Stride(486): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 614)   Stride(486): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(64:U32, 4:U32)U32, 0:U32)U32
 // 	x = 256:U32

 // (op 617) ComputeIndex(614) END
 // (op 617) ComputeIndex(615) BEGIN
 // (op 615) KernelGraph::LoadStoreTileGenerator::ComputeIndex(615): target 42 increment 251 base 485 offset 487 stride 488 buffer -1
 // FastArithmetic:	orig = Add(Multiply(0:I, 64:U32)U32, 1:I)U32
 // 	x = 1:U32

 // (op 615)   Stride(488): indexExpr: 1:U32
 // (op 615)   Stride(488): indexExprPaddingBytes: 0:U32
 // (op 615)   Stride(488): unitStride: true vgprBlockSize: 0
 // (op 615)   Stride(488): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 615)   Stride(488): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(1:U32, 4:U32)U32, 0:U32)U32
 // 	x = 4:U32

 // (op 617) ComputeIndex(615) END
 // (op 617) LoadLDSTile{Value: Float}(37) BEGIN
 // (op 37) GEN: loadMacroTileLDS OP 37 LDS 42MacroTile 8
 // (op 37) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 37) Generate 256:U32 into nullptr
 // (op 37) Generate 4:U32 into nullptr
 // tag 8: v**UNALLOCATED**
 // (op 37) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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

 // (op 37) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: (unallocated)
 // 	info.rowOffsetReg = Offset614: VGPR Value: UInt32 x 1: v7
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // (op 37) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 256
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 16 VGPRs (Value: Float) (op 37): v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23
ds_read_b128 v[8:11], v7 // (op 37) Load local data 
// VMEM: Expected complete at 1548 (current 1528)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
ds_read_b128 v[12:15], v7 offset:256 // (op 37) Load local data 
// VMEM: Expected complete at 1549 (current 1529)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
ds_read_b128 v[16:19], v7 offset:512 // (op 37) Load local data 
// VMEM: Expected complete at 1550 (current 1530)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
ds_read_b128 v[20:23], v7 offset:768 // (op 37) Load local data 
// VMEM: Expected complete at 1551 (current 1531)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 617) LoadLDSTile{Value: Float}(37) END
 // (op 617) end: generate({614})
 // Freeing Offset614: 1 VGPR (Value: UInt32) (op 614): v7

 // Deleting tag 485
 // (op 617) Unlock Scope 617
 // (op 522) Scope(617) END
 // (op 522) Deallocate{}(734) BEGIN
 // (op 734) Deallocate 42
 // Deleting tag 42
 // (op 522) Deallocate{}(734) END
 // (op 522) NOP(625) BEGIN
 // (op 625) generate({})
 // (op 625) end: generate({})
 // (op 522) NOP(625) END
 // (op 522) ComputeIndex(622) BEGIN
 // (op 622) KernelGraph::LoadStoreTileGenerator::ComputeIndex(622): target 47 increment 269 base -1 offset 489 stride 490 buffer 492
 // tag 489: v**UNALLOCATED**
 // FastArithmetic:	orig = {Join: Add(Multiply({Flatten: Add(Multiply({Workgroup Index X: s2:U32}, 64:U32)U32, {Flatten: Add(Multiply({Tile[1]: Modulo({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32})U32}, 1:U64)U64, Multiply({Flatten: Add(Multiply({Workgroup Index Y: s3:U32}, 64:U32)U32, {Flatten: Add(Multiply({Tile[0]: Divide({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32})U32}, Tensor_15_stride_1_14:I64)I64)U64}
 // 	x = {Join: Add(Convert({Flatten: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({Flatten: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Flatten: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32})U32}, Tensor_15_stride_1_14:I64)I64)U64}

 // (op 622)   Offset(489): indexExpr: {Join: Add(Convert({Flatten: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({Flatten: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Flatten: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32})U32}, Tensor_15_stride_1_14:I64)I64)U64}
 // (op 622)   Offset(489): paddingBytes: 0:U32
 // (op 622) Generate Convert(Add(Multiply({Join: Add(Convert({Flatten: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({Flatten: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Flatten: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32})U32}, Tensor_15_stride_1_14:I64)I64)U64}, 4:U32)U64, 0:U32)U64)U32 into Offset622: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Join: Add(Convert({Flatten: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({Flatten: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Flatten: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32})U32}, Tensor_15_stride_1_14:I64)I64)U64}, 4:U32)U64, 0:U32)U64)U32
 // 	x = Convert({Join: AddShiftL(Convert({Flatten: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({Flatten: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Flatten: ShiftL(LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32, 2:U32)U32})U32}, Tensor_15_stride_1_14:I64)I64, 2:U32)U64})U32

 // (op 622) Get arg Tensor_15_stride_1_14
 // (op 622) reg expression
 // (op 622) reg expression
 // (op 622) reg expression
 // (op 622) reg expression
 // (op 622) BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32
 // (op 622) LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 622): v7
v_and_b32 v7, -16, v1 // (op 622) 
// Allocated : 1 VGPR (Value: UInt32) (op 622): v6
v_lshrrev_b32 v6, 4, v1 // (op 622) 
 // (op 622) BEGIN: Tile[1]
 // (op 622) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v7:U32)U32}
 // (op 622) BEGIN: Flatten
 // (op 622) {Flatten: ShiftL(v6:U32, 2:U32)U32}
 // (op 622) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 622)         value (VGPR Value: UInt32 x 1: v6) 
 // (op 622)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 622): v24
v_sub_u32 v24, v1, v7 // (op 622) 
 // (op 622) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 622): v7

// Allocated : 1 VGPR (Value: UInt32) (op 622): v7
v_lshlrev_b32 v7, 2, v6 // (op 622) 
 // (op 622) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 622): v6

 // (op 622) BEGIN: Flatten
 // (op 622) {Flatten: ShiftL({Tile[1]: v24:U32}, 2:U32)U32}
 // (op 622) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 622)         value (Tile[1]: VGPR Value: UInt32 x 1: v24) 
 // (op 622)         shiftAmount (Literal Value: UInt32 x 0: 2)
 // (op 622) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 622): v6
v_lshlrev_b32 v6, 2, v24 // (op 622) 
 // (op 622) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 622): v24

// Allocated : 1 VGPR (Value: UInt32) (op 622): v24
v_lshl_add_u32 v24, s3, 6, v7 // (op 622) 
 // (op 622) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 622): v7

 // (op 622) BEGIN: Flatten
 // (op 622) Multiply({Flatten: v24:U32}, {Tensor_15_stride_1_14: s[30:31]:I64})I64
 // (op 622) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 622)           lhs (Flatten: VGPR Value: UInt32 x 1: v24) 
 // (op 622)           rhs (Tensor_15_stride_1_14: SGPR Value: Int64 x 1: s[30:31])
// Allocated : 1 VGPR (Value: UInt32) (op 622): v7
v_lshl_add_u32 v7, s2, 6, v6 // (op 622) 
 // (op 622) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 622): v6

// Allocated : 2 VGPRs (Value: Int64) (op 622): v26, v27
v_mul_lo_u32 v27, v24, s31 // (op 622) most significant: low of low * high
 // (op 622) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 622): v6
v_mul_hi_u32 v6, v24, s30 // (op 622) most significant: high of low * low
v_mul_lo_u32 v26, v24, s30 // (op 622) least significant: low of low * low
v_add_u32 v27, v27, v6 // (op 622) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 622): v6

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 622): v24

// Allocated Flatten: 2 VGPRs (Value: UInt64) (op 622): v24, v25
v_mov_b32 v25, 0 // (op 622) convert
v_mov_b32 v24, v7 // (op 622) convert
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 622): v7

 // (op 622) BEGIN: Join
 // (op 622) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 622)      lhs (Flatten: VGPR Value: UInt64 x 1: v[24:25]) 
 // (op 622)      rhs (VGPR Value: Int64 x 1: v[26:27])
// Allocated : 2 VGPRs (Value: UInt64) (op 622): v6, v7
// Allocated : 2 SGPRs (Value: Bool64) (op 622): s0, s1
v_add_co_u32 v6, s[0:1], v24, v26 // (op 622) least significant half
v_addc_co_u32 v7, s[0:1], v25, v27, s[0:1] // (op 622) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 622): s0, s1

 // (op 622) ShiftL: dest (VGPR Value: UInt64 x 1: v[6:7]) = 
 // (op 622)         value (VGPR Value: UInt64 x 1: v[6:7]) 
 // (op 622)         shiftAmount (Literal Value: UInt32 x 0: 2)
v_lshlrev_b64 v[6:7], 2, v[6:7] // (op 622) 
 // (op 622) END: Join
 // Freeing : 2 VGPRs (Value: Int64) (op 622): v26, v27

 // Freeing Flatten: 2 VGPRs (Value: UInt64) (op 622): v24, v25

// Allocated Offset622: 1 VGPR (Value: UInt32) (op 622): v24
v_mov_b32 v24, v6 // (op 622) convert
 // Freeing Join: 2 VGPRs (Value: UInt64) (op 622): v6, v7

 // FastArithmetic:	orig = Add(Multiply(0:U32, 1:U64)U64, Multiply(1:U32, Tensor_15_stride_1_14:I64)I64)U64
 // 	x = Convert(Tensor_15_stride_1_14:I64)U64

 // (op 622)   Stride(490): indexExpr: Convert(Tensor_15_stride_1_14:I64)U64
 // (op 622)   Stride(490): indexExprPaddingBytes: 0:U32
 // (op 622)   Stride(490): unitStride: false vgprBlockSize: 0
 // (op 622)   Stride(490): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 622)   Stride(490): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(Convert(Tensor_15_stride_1_14:I64)U64, 4:U32)U64, 0:U32)U64
 // 	x = ShiftL(Convert(Tensor_15_stride_1_14:I64)U64, 2:U32)U64

 // tag 492: s**UNALLOCATED**
 // (op 622) Get arg Tensor_15_pointer
// Allocated Buffer492: 4 SGPRs (Buffer: None) (op 622): s8, s9, s10, s11
s_mov_b32 s8, s34 // (op 622) 
s_mov_b32 s9, s35 // (op 622) 
s_mov_b32 s11, 131072 // (op 622) default options
 // (op 622) Generate Multiply(Tensor_15_extent_15:I64, 4:U32)I64 into nullptr
 // FastArithmetic:	orig = Multiply(Tensor_15_extent_15:I64, 4:U32)I64
 // 	x = ShiftL(Tensor_15_extent_15:I64, 2:U32)I64

 // (op 622) Get arg Tensor_15_extent_15
 // (op 622) reg expression
 // (op 622) ShiftL({Tensor_15_extent_15: s[32:33]:I64}, 2:U32)I64
 // (op 622) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 622)         value (Tensor_15_extent_15: SGPR Value: Int64 x 1: s[32:33]) 
 // (op 622)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 622): s0, s1
s_lshl_b64 s[0:1], s[32:33], 2 // (op 622) 
s_mov_b32 s10, s0 // (op 622) 
 // Freeing : 2 SGPRs (Value: Int64) (op 622): s0, s1

 // (op 522)  Tag 622non referenced 	extraArgs = {Tensor_0_size_0_8, Tensor_2_size_1_12}

 // (op 522) ComputeIndex(622) END
 // (op 522) ComputeIndex(623) BEGIN
 // (op 623) KernelGraph::LoadStoreTileGenerator::ComputeIndex(623): target 47 increment 270 base 489 offset 493 stride 494 buffer -1
 // FastArithmetic:	orig = Add(Multiply(1:U32, 1:U64)U64, Multiply(0:U32, Tensor_15_stride_1_14:I64)I64)U64
 // 	x = 1:U64

 // (op 623)   Stride(494): indexExpr: 1:U64
 // (op 623)   Stride(494): indexExprPaddingBytes: 0:U32
 // (op 623)   Stride(494): unitStride: true vgprBlockSize: 0
 // (op 623)   Stride(494): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 623)   Stride(494): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(1:U64, 4:U32)U64, 0:U32)U64
 // 	x = 4:U64

 // (op 522)  Tag 623non referenced 	extraArgs = {Tensor_0_size_0_8, Tensor_15_extent_15, Tensor_15_stride_1_14, Tensor_2_size_1_12}

 // (op 522) ComputeIndex(623) END
 // (op 522) StoreTiled Value: Float(39) BEGIN
 // (op 39) GEN: storeMacroTileVGPR OP 39 MacroTile 8
 // (op 39) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 39) Generate ShiftL(Convert(Tensor_15_stride_1_14:I64)U64, 2:U32)U64 into VGPR Value: UInt64 x 1: (unallocated)
 // (op 39) Get arg Tensor_15_stride_1_14
 // (op 39) reg expression
 // (op 39) convert
 // (op 39) ShiftL({ convertInPlaceTensor_15_stride_1_14: s[30:31]:U64}, 2:U32)U64
 // (op 39) ShiftL: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 39)         value ( convertInPlaceTensor_15_stride_1_14: SGPR Value: UInt64 x 1: s[30:31]) 
 // (op 39)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 VGPRs (Value: UInt64) (op 39): v6, v7
v_lshlrev_b64 v[6:7], 2, s[30:31] // (op 39) 
 // (op 39) Generate 4:U64 into nullptr
 // (op 39) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: v[8:23]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = VGPR Value: UInt64 x 1: v[6:7]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = Buffer492: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 1 VGPR (Value: UInt32) (op 39): v25
v_mov_b32 v25, v24 // (op 39) 
 // (op 39)   M 4 N 4 elementsPerMove 4 bytesPerMove 16 rowStride v[6:7]:U64 colStride 4:U64 vgprBlockSize 0 numVGPRBlocks 1
s_waitcnt lgkmcnt(3)
buffer_store_dwordx4 v[8:11], v25, s[8:11], 0 offen // (op 39) Store value
// VMEM: Expected complete at 1042 (current 967)
 // (op 39) Generate Add(v25:U32, v6:R)U32 into VGPR Value: UInt32 x 1: v25
 // (op 39) reg expression
 // (op 39) reg expression
 // (op 39) Add(v25:U32, v6:R)U32
s_nop 1
v_add_u32 v25, v25, v6 // (op 39) Wait state hazard: Buffer Store Read Hazard
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[12:15], v25, s[8:11], 0 offen // (op 39) Store value
// VMEM: Expected complete at 1047 (current 972)
 // (op 39) Generate Add(v25:U32, v6:R)U32 into VGPR Value: UInt32 x 1: v25
 // (op 39) reg expression
 // (op 39) reg expression
 // (op 39) Add(v25:U32, v6:R)U32
s_nop 1
v_add_u32 v25, v25, v6 // (op 39) Wait state hazard: Buffer Store Read Hazard
s_waitcnt lgkmcnt(1)
buffer_store_dwordx4 v[16:19], v25, s[8:11], 0 offen // (op 39) Store value
// VMEM: Expected complete at 1052 (current 977)
 // (op 39) Generate Add(v25:U32, v6:R)U32 into VGPR Value: UInt32 x 1: v25
 // (op 39) reg expression
 // (op 39) reg expression
 // (op 39) Add(v25:U32, v6:R)U32
s_nop 1
v_add_u32 v25, v25, v6 // (op 39) Wait state hazard: Buffer Store Read Hazard
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[20:23], v25, s[8:11], 0 offen // (op 39) Store value
// VMEM: Expected complete at 1057 (current 982)
 // Freeing : 2 VGPRs (Value: UInt64) (op 39): v6, v7

 // Freeing : 1 VGPR (Value: UInt32) (op 39): v25

 // (op 522) StoreTiled Value: Float(39) END
 // (op 522) Deallocate{}(656) BEGIN
 // (op 656) Deallocate 8
 // Freeing : 16 VGPRs (Value: Float) (op 37): v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23

 // Deleting tag 8
 // (op 522) Deallocate{}(656) END
 // (op 522) end: generate({525})
 // Freeing Buffer492: 4 SGPRs (Buffer: None) (op 622): s8, s9, s10, s11

 // Deleting tag 492
 // Freeing Offset622: 1 VGPR (Value: UInt32) (op 622): v24

 // Deleting tag 489
 // (op 522) Unlock Scope 522
 // (op 467) Scope(522) END
 // (op 467) end: generate({522})
s_waitcnt vmcnt(63) lgkmcnt(15) expcnt(7)// Keep queues within max waitcnt limit
 // (op 467) 
s_branch GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ConditionalBottom_0_beta_467 // (op 467) Condition: Done, jump to GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ConditionalBottom_0_beta_467
GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ConditionalFalse_0_beta_467:
 // (op 467) 
 // (op 467) generate({528})
 // (op 467) Scope(528) BEGIN
 // (op 528) Lock Scope 528
 // (op 528) generate({531})
 // (op 528) NOP(531) BEGIN
 // (op 531) generate({})
 // (op 531) end: generate({})
 // (op 528) NOP(531) END
 // (op 528) NOP(475) BEGIN
 // (op 475) generate({})
 // (op 475) end: generate({})
 // (op 528) NOP(475) END
 // (op 528) Assign VGPR Multiply(DataFlowTag(34)NA, DataFlowTag(36)NA)NA(476) BEGIN
 // (op 476) Assign dim(423) = Multiply(DataFlowTag(34)NA, DataFlowTag(36)NA)NA
 // (op 476) Generate Multiply(DataFlowTag(34)NA, DataFlowTag(36)NA)NA into nullptr
 // (op 476) reg expression
 // (op 476) reg expression
 // (op 476) Multiply(v5:S, {DataFlowTag36: a[0:15]:S})S
 // (op 528) Scope(583) BEGIN
 // (op 583) Lock Scope 583
 // (op 583) generate({580})
 // (op 583) ComputeIndex(580) BEGIN
 // (op 580) KernelGraph::LoadStoreTileGenerator::ComputeIndex(580): target 3 increment 194 base -1 offset 459 stride 460 buffer 462
 // tag 459: v**UNALLOCATED**
 // FastArithmetic:	orig = {Split: Add(Multiply({Tile: Add(Multiply({Workgroup Index X: s2:U32}, 64:U32)U32, {Tile: Add(Multiply({Tile: Divide({Flatten[0]: Divide({Workitem Index X: v1:U32}, 64:U32)U32}, 2:U32)U32}, 32:U32)U32, {Tile: Add(Multiply({Flatten[0]: Divide({Tile: Add(Multiply(0:U32, 16:U32)U32, {Flatten[0]: Divide({Flatten[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 4:U32)U32})U32}, 8:U32)U32}, 4:I)U32, 0:U32)U32})U32})U32}, 1:U64)U64, Multiply({Tile: Add(Multiply({Workgroup Index Y: s3:U32}, 64:U32)U32, {Tile: Add(Multiply({Tile: Modulo({Flatten[0]: Divide({Workitem Index X: v1:U32}, 64:U32)U32}, 2:U32)U32}, 32:U32)U32, {Tile: Add(Multiply({Flatten[1]: Modulo({Tile: Add(Multiply(0:U32, 16:U32)U32, {Flatten[0]: Divide({Flatten[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 4:U32)U32})U32}, 8:U32)U32}, 4:U32)U32, {Flatten[1]: Modulo({Flatten[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 4:U32)U32})U32})U32})U32}, Tensor_4_stride_1_13:I64)I64)U64}
 // 	x = {Split: Add(Convert({Tile: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Tile: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32})U64, Multiply({Tile: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32})U32}, Tensor_4_stride_1_13:I64)I64)U64}

 // (op 580)   Offset(459): indexExpr: {Split: Add(Convert({Tile: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Tile: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32})U64, Multiply({Tile: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32})U32}, Tensor_4_stride_1_13:I64)I64)U64}
 // (op 580)   Offset(459): paddingBytes: 0:U32
 // (op 580) Generate Convert(Add(Multiply({Split: Add(Convert({Tile: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Tile: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32})U64, Multiply({Tile: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32})U32}, Tensor_4_stride_1_13:I64)I64)U64}, 4:U32)U64, 0:U32)U64)U32 into Offset580: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Split: Add(Convert({Tile: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Tile: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32})U64, Multiply({Tile: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32})U32}, Tensor_4_stride_1_13:I64)I64)U64}, 4:U32)U64, 0:U32)U64)U32
 // 	x = Convert({Split: AddShiftL(Convert({Tile: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Tile: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Tile: ShiftL(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32})U64, Multiply({Tile: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Tile: ShiftLAdd({Tile: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Tile: ShiftLAdd({Flatten[1]: BitwiseAnd(LogicalShiftR({Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32})U32}, Tensor_4_stride_1_13:I64)I64, 2:U32)U64})U32

 // (op 580) Get arg Tensor_4_stride_1_13
 // (op 580) reg expression
 // (op 580) reg expression
 // (op 580) reg expression
 // (op 580) reg expression
 // (op 580) BEGIN: Flatten[1]
 // (op 580) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 580) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 580): v6
v_and_b32 v6, 63, v1 // (op 580) 
 // (op 580) END: Flatten[1]
// Allocated : 1 VGPR (Value: UInt32) (op 580): v7
v_lshrrev_b32 v7, 6, v1 // (op 580) 
 // (op 580) LogicalShiftR({Flatten[1]: v6:U32}, 2:U32)U32
 // (op 580) LogicalShiftR({Flatten[1]: v6:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 580): v8
v_lshrrev_b32 v8, 2, v6 // (op 580) 
// Allocated : 1 VGPR (Value: UInt32) (op 580): v9
v_lshrrev_b32 v9, 5, v6 // (op 580) 
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 580): v6

 // (op 580) BEGIN: Flatten[1]
 // (op 580) {Flatten[1]: BitwiseAnd(v8:U32, 7:U32)U32}
 // (op 580) BEGIN: Flatten[1]
 // (op 580) {Flatten[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 580): v6
v_and_b32 v6, 7, v8 // (op 580) 
 // (op 580) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 580): v8

// Allocated : 1 VGPR (Value: UInt32) (op 580): v8
v_and_b32 v8, 3, v1 // (op 580) 
 // (op 580) END: Flatten[1]
 // (op 580) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 580) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 580): v10
v_lshrrev_b32 v10, 7, v1 // (op 580) 
// Allocated : 1 VGPR (Value: UInt32) (op 580): v11
v_lshl_add_u32 v11, v6, 2, v8 // (op 580) 
 // (op 580) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 580): v8

 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 580): v6

 // (op 580) BEGIN: Tile
 // (op 580) {Tile: ShiftL(v9:U32, 2:U32)U32}
 // (op 580) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 580)         value (VGPR Value: UInt32 x 1: v9) 
 // (op 580)         shiftAmount (Literal Value: UInt32 x 0: 2)
 // (op 580) BEGIN: Tile
 // (op 580) {Tile: BitwiseAnd(v7:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 580): v6
v_lshlrev_b32 v6, 2, v9 // (op 580) 
 // (op 580) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 580): v9

// Allocated : 1 VGPR (Value: UInt32) (op 580): v9
v_and_b32 v9, 1, v7 // (op 580) 
 // (op 580) END: Tile
 // Freeing : 1 VGPR (Value: UInt32) (op 580): v7

 // (op 580) BEGIN: Tile
 // (op 580) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 580): v8
v_lshl_add_u32 v8, v10, 5, v6 // (op 580) 
 // (op 580) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 580): v6

 // Freeing : 1 VGPR (Value: UInt32) (op 580): v10

// Allocated : 1 VGPR (Value: UInt32) (op 580): v10
v_lshl_add_u32 v10, v9, 5, v11 // (op 580) 
 // (op 580) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 580): v11

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 580): v9

 // (op 580) BEGIN: Tile
 // (op 580) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 580): v9
v_lshl_add_u32 v9, s2, 6, v8 // (op 580) 
 // (op 580) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 580): v8

// Allocated : 1 VGPR (Value: UInt32) (op 580): v8
v_lshl_add_u32 v8, s3, 6, v10 // (op 580) 
 // (op 580) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 580): v10

// Allocated Tile: 2 VGPRs (Value: UInt64) (op 580): v6, v7
v_mov_b32 v7, 0 // (op 580) convert
 // (op 580) Multiply({Tile: v8:U32}, {Tensor_4_stride_1_13: s[28:29]:I64})I64
 // (op 580) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 580)           lhs (Tile: VGPR Value: UInt32 x 1: v8) 
 // (op 580)           rhs (Tensor_4_stride_1_13: SGPR Value: Int64 x 1: s[28:29])
v_mov_b32 v6, v9 // (op 580) convert
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 580): v9

// Allocated : 2 VGPRs (Value: Int64) (op 580): v10, v11
v_mul_lo_u32 v11, v8, s29 // (op 580) most significant: low of low * high
 // (op 580) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 580): v9
v_mul_hi_u32 v9, v8, s28 // (op 580) most significant: high of low * low
v_mul_lo_u32 v10, v8, s28 // (op 580) least significant: low of low * low
v_add_u32 v11, v11, v9 // (op 580) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 580): v9

 // Freeing Tile: 1 VGPR (Value: UInt32) (op 580): v8

 // (op 580) BEGIN: Split
 // (op 580) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 580)      lhs (Tile: VGPR Value: UInt64 x 1: v[6:7]) 
 // (op 580)      rhs (VGPR Value: Int64 x 1: v[10:11])
// Allocated : 2 VGPRs (Value: UInt64) (op 580): v8, v9
// Allocated : 2 SGPRs (Value: Bool64) (op 580): s0, s1
v_add_co_u32 v8, s[0:1], v6, v10 // (op 580) least significant half
v_addc_co_u32 v9, s[0:1], v7, v11, s[0:1] // (op 580) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 580): s0, s1

 // (op 580) ShiftL: dest (VGPR Value: UInt64 x 1: v[8:9]) = 
 // (op 580)         value (VGPR Value: UInt64 x 1: v[8:9]) 
 // (op 580)         shiftAmount (Literal Value: UInt32 x 0: 2)
v_lshlrev_b64 v[8:9], 2, v[8:9] // (op 580) 
 // (op 580) END: Split
 // Freeing : 2 VGPRs (Value: Int64) (op 580): v10, v11

 // Freeing Tile: 2 VGPRs (Value: UInt64) (op 580): v6, v7

// Allocated Offset580: 1 VGPR (Value: UInt32) (op 580): v7
v_mov_b32 v7, v8 // (op 580) convert
 // Freeing Split: 2 VGPRs (Value: UInt64) (op 580): v8, v9

 // FastArithmetic:	orig = Add(Multiply(8:U32, 1:U64)U64, Multiply(0:U32, Tensor_4_stride_1_13:I64)I64)U64
 // 	x = 8:U64

 // (op 580)   Stride(460): indexExpr: 8:U64
 // (op 580)   Stride(460): indexExprPaddingBytes: 0:U32
 // (op 580)   Stride(460): unitStride: false vgprBlockSize: 0
 // (op 580)   Stride(460): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 580)   Stride(460): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(8:U64, 4:U32)U64, 0:U32)U64
 // 	x = 32:U64

 // tag 462: s**UNALLOCATED**
 // (op 580) Get arg Tensor_4_pointer
// Allocated Buffer462: 4 SGPRs (Buffer: None) (op 580): s8, s9, s10, s11
s_mov_b32 s8, s14 // (op 580) 
s_mov_b32 s9, s15 // (op 580) 
s_mov_b32 s11, 131072 // (op 580) default options
 // (op 580) Generate Multiply(Tensor_4_extent_4:I64, 4:U32)I64 into nullptr
 // FastArithmetic:	orig = Multiply(Tensor_4_extent_4:I64, 4:U32)I64
 // 	x = ShiftL(Tensor_4_extent_4:I64, 2:U32)I64

 // (op 580) Get arg Tensor_4_extent_4
 // (op 580) reg expression
 // (op 580) ShiftL({Tensor_4_extent_4: s[12:13]:I64}, 2:U32)I64
 // (op 580) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 580)         value (Tensor_4_extent_4: SGPR Value: Int64 x 1: s[12:13]) 
 // (op 580)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 580): s0, s1
s_lshl_b64 s[0:1], s[12:13], 2 // (op 580) 
s_mov_b32 s10, s0 // (op 580) 
 // Freeing : 2 SGPRs (Value: Int64) (op 580): s0, s1

 // (op 583)  Tag 580non referenced 	extraArgs = {Tensor_0_size_0_8, Tensor_2_size_1_12}

 // (op 583) ComputeIndex(580) END
 // (op 583) Deallocate{Tensor_4_pointer}(802) BEGIN
 // (op 802) Deallocate Tensor_4_pointer
 // Freeing Tensor_4_pointer: 2 SGPRs (Value: Raw32): s14, s15

 // (op 583) Deallocate{Tensor_4_pointer}(802) END
 // (op 583) ComputeIndex(581) BEGIN
 // (op 581) KernelGraph::LoadStoreTileGenerator::ComputeIndex(581): target 3 increment 195 base 459 offset 463 stride 464 buffer -1
 // FastArithmetic:	orig = Add(Multiply(1:U32, 1:U64)U64, Multiply(0:U32, Tensor_4_stride_1_13:I64)I64)U64
 // 	x = 1:U64

 // (op 581)   Stride(464): indexExpr: 1:U64
 // (op 581)   Stride(464): indexExprPaddingBytes: 0:U32
 // (op 581)   Stride(464): unitStride: true vgprBlockSize: 0
 // (op 581)   Stride(464): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 581)   Stride(464): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(1:U64, 4:U32)U64, 0:U32)U64
 // 	x = 4:U64

 // (op 583)  Tag 581non referenced 	extraArgs = {Tensor_0_size_0_8, Tensor_2_size_1_12, Tensor_4_extent_4, Tensor_4_stride_1_13}

 // (op 583) ComputeIndex(581) END
 // (op 583) Deallocate{Tensor_4_extent_4}(804) BEGIN
 // (op 804) Deallocate Tensor_4_extent_4
 // Freeing Tensor_4_extent_4: 2 SGPRs (Value: Raw32): s12, s13

 // (op 583) Deallocate{Tensor_4_extent_4}(804) END
 // (op 583) LoadTiled Value: Float(481) BEGIN
 // (op 481) GEN: loadMacroTileWAVECIACCUM OP 481 WaveTile 177
 // (op 481) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 481) Generate 32:U64 into nullptr
 // (op 481) Generate 4:U64 into nullptr
 // tag 419: v**UNALLOCATED**
 // (op 481) 	Dir = Load
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
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = Buffer462: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 1
 // 	colStrideIsOne = 1

 // (op 481) LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: (unallocated)
 // 	info.rowOffsetReg = Offset580Split: VGPR Value: UInt32 x 1: v7
 // 	info.rowStrideReg = Literal Value: UInt64 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = Buffer462: SGPR Buffer: None x 1: s[8:11]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }
 // (op 481) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 32
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 16 VGPRs (Value: Float) (op 481): v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23
buffer_load_dwordx4 v[8:11], v7, s[8:11], 0 offen // (op 481) Load value
// VMEM: Expected stall of 29. CBNW: 0, Inc: 4
// VMEM: Expected complete at 1118 (current 1043)
buffer_load_dwordx4 v[12:15], v7, s[8:11], 0 offen offset:32 // (op 481) Load value
// VMEM: Expected stall of 4. CBNW: 1, Inc: 4
// VMEM: Expected complete at 1123 (current 1048)
buffer_load_dwordx4 v[16:19], v7, s[8:11], 0 offen offset:64 // (op 481) Load value
// VMEM: Expected stall of 4. CBNW: 2, Inc: 4
// VMEM: Expected complete at 1128 (current 1053)
buffer_load_dwordx4 v[20:23], v7, s[8:11], 0 offen offset:96 // (op 481) Load value
// VMEM: Expected stall of 4. CBNW: 3, Inc: 4
// VMEM: Expected complete at 1133 (current 1058)
 // (op 583)  Tag 481non referenced 	extraArgs = {Tensor_4_stride_1_13}

 // (op 583) LoadTiled Value: Float(481) END
 // (op 583) Deallocate{Tensor_4_stride_1_13}(794) BEGIN
 // (op 794) Deallocate Tensor_4_stride_1_13
 // Freeing Tensor_4_stride_1_13: 2 SGPRs (Value: Raw32): s28, s29

 // (op 583) Deallocate{Tensor_4_stride_1_13}(794) END
 // (op 583) end: generate({580})
 // Freeing Buffer462: 4 SGPRs (Buffer: None) (op 580): s8, s9, s10, s11

 // Deleting tag 462
 // Freeing Offset580: 1 VGPR (Value: UInt32) (op 580): v7

 // Deleting tag 459
 // (op 583) Unlock Scope 583
 // (op 528) Scope(583) END
// Allocated : 16 VGPRs (Value: Float) (op 476): v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39
v_accvgpr_read v24, a0 // (op 476) 
 // (op 528) Assign VGPR Multiply(DataFlowTag(25)NA, DataFlowTag(419)NA)NA(482) BEGIN
 // (op 482) Assign dim(421) = Multiply(DataFlowTag(25)NA, DataFlowTag(419)NA)NA
 // (op 482) Generate Multiply(DataFlowTag(25)NA, DataFlowTag(419)NA)NA into nullptr
 // (op 482) reg expression
 // (op 482) reg expression
 // (op 482) Multiply(v3:S, v[8:23]:S)S
v_accvgpr_read v25, a1 // (op 476) 
v_accvgpr_read v26, a2 // (op 476) 
v_accvgpr_read v27, a3 // (op 476) 
v_accvgpr_read v28, a4 // (op 476) 
v_accvgpr_read v29, a5 // (op 476) 
v_accvgpr_read v30, a6 // (op 476) 
v_accvgpr_read v31, a7 // (op 476) 
v_accvgpr_read v32, a8 // (op 476) 
v_accvgpr_read v33, a9 // (op 476) 
v_accvgpr_read v34, a10 // (op 476) 
v_accvgpr_read v35, a11 // (op 476) 
v_accvgpr_read v36, a12 // (op 476) 
v_accvgpr_read v37, a13 // (op 476) 
v_accvgpr_read v38, a14 // (op 476) 
v_accvgpr_read v39, a15 // (op 476) 
// Allocated : 16 VGPRs (Value: Float) (op 476): v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55
v_mul_f32 v40, v5, v24 // (op 476) 
v_mul_f32 v41, v5, v25 // (op 476) 
v_mul_f32 v42, v5, v26 // (op 476) 
v_mul_f32 v43, v5, v27 // (op 476) 
v_mul_f32 v44, v5, v28 // (op 476) 
v_mul_f32 v45, v5, v29 // (op 476) 
v_mul_f32 v46, v5, v30 // (op 476) 
v_mul_f32 v47, v5, v31 // (op 476) 
v_mul_f32 v48, v5, v32 // (op 476) 
v_mul_f32 v49, v5, v33 // (op 476) 
v_mul_f32 v50, v5, v34 // (op 476) 
v_mul_f32 v51, v5, v35 // (op 476) 
v_mul_f32 v52, v5, v36 // (op 476) 
v_mul_f32 v53, v5, v37 // (op 476) 
v_mul_f32 v54, v5, v38 // (op 476) 
v_mul_f32 v55, v5, v39 // (op 476) 
 // Freeing : 16 VGPRs (Value: Float) (op 476): v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39

 // (op 528) Assign VGPR Multiply(DataFlowTag(34)NA, DataFlowTag(36)NA)NA(476) END
// Allocated : 16 VGPRs (Value: Float) (op 482): v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39
s_waitcnt vmcnt(3)
v_mul_f32 v24, v3, v8 // (op 482) 
v_mul_f32 v25, v3, v9 // (op 482) 
v_mul_f32 v26, v3, v10 // (op 482) 
v_mul_f32 v27, v3, v11 // (op 482) 
s_waitcnt vmcnt(2)
v_mul_f32 v28, v3, v12 // (op 482) 
v_mul_f32 v29, v3, v13 // (op 482) 
v_mul_f32 v30, v3, v14 // (op 482) 
v_mul_f32 v31, v3, v15 // (op 482) 
s_waitcnt vmcnt(1)
v_mul_f32 v32, v3, v16 // (op 482) 
v_mul_f32 v33, v3, v17 // (op 482) 
v_mul_f32 v34, v3, v18 // (op 482) 
v_mul_f32 v35, v3, v19 // (op 482) 
s_waitcnt vmcnt(0)
v_mul_f32 v36, v3, v20 // (op 482) 
v_mul_f32 v37, v3, v21 // (op 482) 
v_mul_f32 v38, v3, v22 // (op 482) 
v_mul_f32 v39, v3, v23 // (op 482) 
 // (op 528) Assign VGPR Multiply(DataFlowTag(25)NA, DataFlowTag(419)NA)NA(482) END
 // (op 528) Deallocate{}(730) BEGIN
 // (op 730) Deallocate 419
 // Freeing : 16 VGPRs (Value: Float) (op 481): v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23

 // Deleting tag 419
 // (op 528) Deallocate{}(730) END
 // (op 528) Assign VGPR Add(DataFlowTag(421)NA, DataFlowTag(423)NA)NA(477) BEGIN
 // (op 477) Assign dim(425) = Add(DataFlowTag(421)NA, DataFlowTag(423)NA)NA
 // (op 477) Generate Add(DataFlowTag(421)NA, DataFlowTag(423)NA)NA into nullptr
 // (op 477) reg expression
 // (op 477) reg expression
 // (op 477) Add({DataFlowTag421: v[24:39]:S}, {DataFlowTag423: v[40:55]:S})S
// Allocated : 16 VGPRs (Value: Float) (op 477): v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23
v_add_f32 v8, v24, v40 // (op 477) 
v_add_f32 v9, v25, v41 // (op 477) 
v_add_f32 v10, v26, v42 // (op 477) 
v_add_f32 v11, v27, v43 // (op 477) 
v_add_f32 v12, v28, v44 // (op 477) 
v_add_f32 v13, v29, v45 // (op 477) 
v_add_f32 v14, v30, v46 // (op 477) 
v_add_f32 v15, v31, v47 // (op 477) 
v_add_f32 v16, v32, v48 // (op 477) 
v_add_f32 v17, v33, v49 // (op 477) 
v_add_f32 v18, v34, v50 // (op 477) 
v_add_f32 v19, v35, v51 // (op 477) 
v_add_f32 v20, v36, v52 // (op 477) 
v_add_f32 v21, v37, v53 // (op 477) 
v_add_f32 v22, v38, v54 // (op 477) 
v_add_f32 v23, v39, v55 // (op 477) 
 // (op 528) Assign VGPR Add(DataFlowTag(421)NA, DataFlowTag(423)NA)NA(477) END
 // (op 528) Deallocate{}(726) BEGIN
 // (op 726) Deallocate 421
 // Freeing DataFlowTag421: 16 VGPRs (Value: Float) (op 482): v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37, v38, v39

 // Deleting tag 421
 // (op 726) Deallocate 423
 // Freeing DataFlowTag423: 16 VGPRs (Value: Float) (op 476): v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55

 // Deleting tag 423
 // (op 528) Deallocate{}(726) END
 // (op 528) Barrier(554) BEGIN
s_barrier  // (op 554) 
 // (op 528) Barrier(554) END
 // (op 528) Scope(639) BEGIN
 // (op 639) Lock Scope 639
 // (op 639) generate({636})
 // (op 639) ComputeIndex(636) BEGIN
 // (op 636) KernelGraph::LoadStoreTileGenerator::ComputeIndex(636): target 427 increment 218 base -1 offset 499 stride 500 buffer -1
 // tag 499: v**UNALLOCATED**
 // FastArithmetic:	orig = {Flatten: Add(Multiply({Flatten: Add(Multiply({Tile[1]: Modulo({Tile[0]: Divide({Workitem Index X: v1:U32}, 64:U32)U32}, 2:U32)U32}, 32:U32)U32, {Flatten: Add(Multiply({Tile[1]: Modulo({Flatten: Add(Multiply(0:U32, 16:U32)U32, {Tile[0]: Divide({Tile[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 4:U32)U32})U32}, 8:U32)U32}, 4:U32)U32, {Tile[1]: Modulo({Tile[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 4:U32)U32})U32})U32}, 64:U32)U32, {Flatten: Add(Multiply({Tile[0]: Divide({Tile[0]: Divide({Workitem Index X: v1:U32}, 64:U32)U32}, 2:U32)U32}, 32:U32)U32, {Flatten: Add(Multiply({Tile[0]: Divide({Flatten: Add(Multiply(0:U32, 16:U32)U32, {Tile[0]: Divide({Tile[1]: Modulo({Workitem Index X: v1:U32}, 64:U32)U32}, 4:U32)U32})U32}, 8:U32)U32}, 4:I)U32, 0:U32)U32})U32})U32}
 // 	x = {Flatten: ShiftLAdd({Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32}, 6:U32, {Flatten: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}

 // (op 636)   Offset(499): indexExpr: {Flatten: ShiftLAdd({Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32}, 6:U32, {Flatten: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}
 // (op 636)   Offset(499): paddingBytes: 0:U32
 // (op 636) Generate Convert(Add(Multiply({Flatten: ShiftLAdd({Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32}, 6:U32, {Flatten: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, 4:U32)U32, 0:U32)U32)U32 into Offset636: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Flatten: ShiftLAdd({Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32}, 6:U32, {Flatten: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, 4:U32)U32, 0:U32)U32)U32
 // 	x = ShiftL({Flatten: ShiftLAdd({Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32, 1:U32)U32}, 5:U32, {Flatten: ShiftLAdd({Tile[1]: BitwiseAnd(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 2:U32)U32, 7:U32)U32}, 2:U32, {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32})U32})U32}, 6:U32, {Flatten: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32, 5:U32, {Flatten: ShiftL(LogicalShiftR({Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}, 5:U32)U32, 2:U32)U32})U32})U32}, 2:U32)U32

 // (op 636) reg expression
 // (op 636) BEGIN: Tile[1]
 // (op 636) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 63:U32)U32}
 // (op 636) LogicalShiftR({Workitem Index X: v1:U32}, 6:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 636): v7
v_and_b32 v7, 63, v1 // (op 636) 
 // (op 636) END: Tile[1]
// Allocated : 1 VGPR (Value: UInt32) (op 636): v6
v_lshrrev_b32 v6, 6, v1 // (op 636) 
 // (op 636) LogicalShiftR({Tile[1]: v7:U32}, 2:U32)U32
 // (op 636) BEGIN: Tile[1]
 // (op 636) {Tile[1]: BitwiseAnd({Workitem Index X: v1:U32}, 3:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 636): v24
v_lshrrev_b32 v24, 2, v7 // (op 636) 
// Allocated : 1 VGPR (Value: UInt32) (op 636): v25
v_and_b32 v25, 3, v1 // (op 636) 
 // (op 636) END: Tile[1]
 // (op 636) BEGIN: Tile[1]
 // (op 636) {Tile[1]: BitwiseAnd(v24:U32, 7:U32)U32}
 // (op 636) LogicalShiftR({Tile[1]: v7:U32}, 5:U32)U32
// Allocated : 1 VGPR (Value: UInt32) (op 636): v26
v_and_b32 v26, 7, v24 // (op 636) 
 // (op 636) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 636): v24

// Allocated : 1 VGPR (Value: UInt32) (op 636): v24
v_lshrrev_b32 v24, 5, v7 // (op 636) 
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 636): v7

 // (op 636) BEGIN: Flatten
 // (op 636) BEGIN: Tile[1]
 // (op 636) {Tile[1]: BitwiseAnd(v6:U32, 1:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 636): v7
v_lshl_add_u32 v7, v26, 2, v25 // (op 636) 
 // (op 636) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 636): v25

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 636): v26

// Allocated : 1 VGPR (Value: UInt32) (op 636): v25
v_and_b32 v25, 1, v6 // (op 636) 
 // (op 636) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 636): v6

 // (op 636) LogicalShiftR({Workitem Index X: v1:U32}, 7:U32)U32
 // (op 636) BEGIN: Flatten
 // (op 636) {Flatten: ShiftL(v24:U32, 2:U32)U32}
 // (op 636) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 636)         value (VGPR Value: UInt32 x 1: v24) 
 // (op 636)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 636): v6
v_lshrrev_b32 v6, 7, v1 // (op 636) 
// Allocated : 1 VGPR (Value: UInt32) (op 636): v26
v_lshlrev_b32 v26, 2, v24 // (op 636) 
 // (op 636) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 636): v24

 // (op 636) BEGIN: Flatten
 // (op 636) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 636): v24
v_lshl_add_u32 v24, v25, 5, v7 // (op 636) 
 // (op 636) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 636): v7

 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 636): v25

// Allocated : 1 VGPR (Value: UInt32) (op 636): v7
v_lshl_add_u32 v7, v6, 5, v26 // (op 636) 
 // (op 636) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 636): v26

 // Freeing : 1 VGPR (Value: UInt32) (op 636): v6

 // (op 636) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 636): v6
v_lshl_add_u32 v6, v24, 6, v7 // (op 636) 
 // (op 636) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 636): v7

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 636): v24

 // (op 636) ShiftL({Flatten: v6:U32}, 2:U32)U32
 // (op 636) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 636)         value (Flatten: VGPR Value: UInt32 x 1: v6) 
 // (op 636)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated Offset636: 1 VGPR (Value: UInt32) (op 636): v7
v_lshlrev_b32 v7, 2, v6 // (op 636) 
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 636): v6

 // FastArithmetic:	orig = Add(Multiply(0:U32, Multiply(1:I, 64:U32)U32)U32, Multiply(8:U32, 1:I)U32)U32
 // 	x = 8:U32

 // (op 636)   Stride(500): indexExpr: 8:U32
 // (op 636)   Stride(500): indexExprPaddingBytes: 0:U32
 // (op 636)   Stride(500): unitStride: false vgprBlockSize: 0
 // (op 636)   Stride(500): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 636)   Stride(500): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(8:U32, 4:U32)U32, 0:U32)U32
 // 	x = 32:U32

 // (op 639) ComputeIndex(636) END
 // (op 639) ComputeIndex(637) BEGIN
 // (op 637) KernelGraph::LoadStoreTileGenerator::ComputeIndex(637): target 427 increment 219 base 499 offset 501 stride 502 buffer -1
 // FastArithmetic:	orig = Add(Multiply(0:U32, Multiply(1:I, 64:U32)U32)U32, Multiply(1:U32, 1:I)U32)U32
 // 	x = 1:U32

 // (op 637)   Stride(502): indexExpr: 1:U32
 // (op 637)   Stride(502): indexExprPaddingBytes: 0:U32
 // (op 637)   Stride(502): unitStride: true vgprBlockSize: 0
 // (op 637)   Stride(502): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 637)   Stride(502): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(1:U32, 4:U32)U32, 0:U32)U32
 // 	x = 4:U32

 // (op 639) ComputeIndex(637) END
 // (op 639) StoreLDSTile Value: Float(478) BEGIN
 // (op 478) GEN: StoreLDSTile
 // (op 478) GEN: storeMacroTileWAVELDS OP 478 LDS 427 MacroTile 425 WaveTile 213
 // (op 478) 	waveTile = WaveTile{1024:I}
 // 	waveTileNumElements = 1024
 // 	activeLanesInWave = 64
 // 	packing = 1
 // 	activeLanesInWave * packing = 64
 // 	store.varType = Value: Float

 // (op 478) 	agpr->description() = DataFlowTag425: VGPR Value: Float x 16: v[8:23]
 // 	waveTile = WaveTile{1024:I}
 // 	waveTileNumElements = 1024
 // 	activeLanesInWave = 64
 // 	packing = 1
 // 	activeLanesInWave * packing = 64

 // (op 478) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 478) Generate 32:U32 into nullptr
 // (op 478) Generate 4:U32 into nullptr
 // (op 478) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag425: VGPR Value: Float x 16: v[8:23]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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

 // (op 478) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = DataFlowTag425: VGPR Value: Float x 16: v[8:23]
 // 	info.rowOffsetReg = Offset636: VGPR Value: UInt32 x 1: v7
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 32
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // (op 478) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 32
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

ds_write_b128 v7, v[8:11] // (op 478) Store local data 
// VMEM: Expected complete at 1752 (current 1732)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
ds_write_b128 v7, v[12:15] offset:32 // (op 478) Store local data 
// VMEM: Expected complete at 1753 (current 1733)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
ds_write_b128 v7, v[16:19] offset:64 // (op 478) Store local data 
// VMEM: Expected complete at 1754 (current 1734)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
ds_write_b128 v7, v[20:23] offset:96 // (op 478) Store local data 
// VMEM: Expected complete at 1755 (current 1735)
// Extra dsts: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 639) StoreLDSTile Value: Float(478) END
 // (op 639) end: generate({636})
 // Freeing Offset636: 1 VGPR (Value: UInt32) (op 636): v7

 // Deleting tag 499
 // (op 639) Unlock Scope 639
 // (op 528) Scope(639) END
 // (op 528) Deallocate{}(736) BEGIN
 // (op 736) Deallocate 425
 // Freeing DataFlowTag425: 16 VGPRs (Value: Float) (op 477): v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23

 // Deleting tag 425
 // (op 528) Deallocate{}(736) END
 // (op 528) Barrier(555) BEGIN
s_waitcnt lgkmcnt(0)
s_barrier  // (op 555) 
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 528) Barrier(555) END
 // (op 528) Scope(647) BEGIN
 // (op 647) Lock Scope 647
 // (op 647) generate({644})
 // (op 647) ComputeIndex(644) BEGIN
 // (op 644) KernelGraph::LoadStoreTileGenerator::ComputeIndex(644): target 427 increment 250 base -1 offset 503 stride 504 buffer -1
 // tag 503: v**UNALLOCATED**
 // FastArithmetic:	orig = {Tile: Add(Multiply({Tile: Add(Multiply({Flatten[0]: Divide({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32}, 64:U32)U32, {Tile: Add(Multiply({Flatten[1]: Modulo({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32})U32}
 // 	x = {Tile: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}

 // (op 644)   Offset(503): indexExpr: {Tile: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}
 // (op 644)   Offset(503): paddingBytes: 0:U32
 // (op 644) Generate Convert(Add(Multiply({Tile: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}, 4:U32)U32, 0:U32)U32)U32 into Offset644: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Tile: ShiftLAdd(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32}, 4:U32)U32, 0:U32)U32)U32
 // 	x = ShiftL({Tile: ShiftLAdd(LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32, 8:U32, {Tile: ShiftL({Flatten[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32)U32}, 2:U32)U32})U32}, 2:U32)U32

 // (op 644) reg expression
 // (op 644) BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32
 // (op 644) LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 644): v6
v_and_b32 v6, -16, v1 // (op 644) 
// Allocated : 1 VGPR (Value: UInt32) (op 644): v7
v_lshrrev_b32 v7, 4, v1 // (op 644) 
 // (op 644) BEGIN: Flatten[1]
 // (op 644) {Flatten[1]: Subtract({Workitem Index X: v1:U32}, v6:U32)U32}
// Allocated : 1 VGPR (Value: UInt32) (op 644): v8
v_sub_u32 v8, v1, v6 // (op 644) 
 // (op 644) END: Flatten[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 644): v6

 // (op 644) BEGIN: Tile
 // (op 644) {Tile: ShiftL({Flatten[1]: v8:U32}, 2:U32)U32}
 // (op 644) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 644)         value (Flatten[1]: VGPR Value: UInt32 x 1: v8) 
 // (op 644)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 644): v6
v_lshlrev_b32 v6, 2, v8 // (op 644) 
 // (op 644) END: Tile
 // Freeing Flatten[1]: 1 VGPR (Value: UInt32) (op 644): v8

 // (op 644) BEGIN: Tile
// Allocated : 1 VGPR (Value: UInt32) (op 644): v8
v_lshl_add_u32 v8, v7, 8, v6 // (op 644) 
 // (op 644) END: Tile
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 644): v6

 // Freeing : 1 VGPR (Value: UInt32) (op 644): v7

 // (op 644) ShiftL({Tile: v8:U32}, 2:U32)U32
 // (op 644) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 644)         value (Tile: VGPR Value: UInt32 x 1: v8) 
 // (op 644)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated Offset644: 1 VGPR (Value: UInt32) (op 644): v7
v_lshlrev_b32 v7, 2, v8 // (op 644) 
 // Freeing Tile: 1 VGPR (Value: UInt32) (op 644): v8

 // FastArithmetic:	orig = Add(Multiply(1:I, 64:U32)U32, 0:I)U32
 // 	x = 64:U32

 // (op 644)   Stride(504): indexExpr: 64:U32
 // (op 644)   Stride(504): indexExprPaddingBytes: 0:U32
 // (op 644)   Stride(504): unitStride: false vgprBlockSize: 0
 // (op 644)   Stride(504): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 644)   Stride(504): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(64:U32, 4:U32)U32, 0:U32)U32
 // 	x = 256:U32

 // (op 647) ComputeIndex(644) END
 // (op 647) ComputeIndex(645) BEGIN
 // (op 645) KernelGraph::LoadStoreTileGenerator::ComputeIndex(645): target 427 increment 251 base 503 offset 505 stride 506 buffer -1
 // FastArithmetic:	orig = Add(Multiply(0:I, 64:U32)U32, 1:I)U32
 // 	x = 1:U32

 // (op 645)   Stride(506): indexExpr: 1:U32
 // (op 645)   Stride(506): indexExprPaddingBytes: 0:U32
 // (op 645)   Stride(506): unitStride: true vgprBlockSize: 0
 // (op 645)   Stride(506): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 645)   Stride(506): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(1:U32, 4:U32)U32, 0:U32)U32
 // 	x = 4:U32

 // (op 647) ComputeIndex(645) END
 // (op 647) LoadLDSTile{Value: Float}(479) BEGIN
 // (op 479) GEN: loadMacroTileLDS OP 479 LDS 427MacroTile 429
 // (op 479) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 479) Generate 256:U32 into nullptr
 // (op 479) Generate 4:U32 into nullptr
 // tag 429: v**UNALLOCATED**
 // (op 479) 	Dir = Load
 // LSTInfo {
 // 	info.kind = Local
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: (unallocated)
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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

 // (op 479) LSTInfo {
 // 	info.kind = Local
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: (unallocated)
 // 	info.rowOffsetReg = Offset644: VGPR Value: UInt32 x 1: v7
 // 	info.rowStrideReg = Literal Value: UInt32 x 0: 256
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt32 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
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
 // (op 479) 	info.m = 4
 // 	info.n = 4
 // 	elementsPerMove = 4
 // 	bytesPerMove = 16
 // 	rowStride = 256
 // 	colStride = 4
 // 	info.colStrideAttributes.elementBlockSize = 0
 // 	numVGPRBlocks = 1
 // 	elementBlockStride = 0

// Allocated : 16 VGPRs (Value: Float) (op 479): v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23
ds_read_b128 v[8:11], v7 // (op 479) Load local data 
// VMEM: Expected complete at 1784 (current 1764)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
ds_read_b128 v[12:15], v7 offset:256 // (op 479) Load local data 
// VMEM: Expected complete at 1785 (current 1765)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
ds_read_b128 v[16:19], v7 offset:512 // (op 479) Load local data 
// VMEM: Expected complete at 1786 (current 1766)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
ds_read_b128 v[20:23], v7 offset:768 // (op 479) Load local data 
// VMEM: Expected complete at 1787 (current 1767)
// Extra srcs: LDS Value: Float x 0: LDS:(Offset: 0, Size: 16384)
 // (op 647) LoadLDSTile{Value: Float}(479) END
 // (op 647) end: generate({644})
 // Freeing Offset644: 1 VGPR (Value: UInt32) (op 644): v7

 // Deleting tag 503
 // (op 647) Unlock Scope 647
 // (op 528) Scope(647) END
 // (op 528) Deallocate{}(738) BEGIN
 // (op 738) Deallocate 427
 // Deleting tag 427
 // (op 528) Deallocate{}(738) END
 // (op 528) NOP(632) BEGIN
 // (op 632) generate({})
 // (op 632) end: generate({})
 // (op 528) NOP(632) END
 // (op 528) ComputeIndex(629) BEGIN
 // (op 629) KernelGraph::LoadStoreTileGenerator::ComputeIndex(629): target 47 increment 269 base -1 offset 495 stride 496 buffer 492
 // tag 495: v**UNALLOCATED**
 // FastArithmetic:	orig = {Join: Add(Multiply({Flatten: Add(Multiply({Workgroup Index X: s2:U32}, 64:U32)U32, {Flatten: Add(Multiply({Tile[1]: Modulo({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32})U32}, 1:U64)U64, Multiply({Flatten: Add(Multiply({Workgroup Index Y: s3:U32}, 64:U32)U32, {Flatten: Add(Multiply({Tile[0]: Divide({Workitem Index X: v1:U32}, 16:I)U32}, 4:I)U32, 0:U32)U32})U32}, Tensor_15_stride_1_14:I64)I64)U64}
 // 	x = {Join: Add(Convert({Flatten: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({Flatten: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Flatten: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32})U32}, Tensor_15_stride_1_14:I64)I64)U64}

 // (op 629)   Offset(495): indexExpr: {Join: Add(Convert({Flatten: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({Flatten: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Flatten: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32})U32}, Tensor_15_stride_1_14:I64)I64)U64}
 // (op 629)   Offset(495): paddingBytes: 0:U32
 // (op 629) Generate Convert(Add(Multiply({Join: Add(Convert({Flatten: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({Flatten: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Flatten: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32})U32}, Tensor_15_stride_1_14:I64)I64)U64}, 4:U32)U64, 0:U32)U64)U32 into Offset629: VGPR Value: UInt32 x 1: (unallocated)
 // FastArithmetic:	orig = Convert(Add(Multiply({Join: Add(Convert({Flatten: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({Flatten: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Flatten: ShiftL(LogicalShiftR(Add({Workitem Index X: v1:U32}, LogicalShiftR({Workitem Index X: v1:U32}, 59:U32)U32)U32, 4:I)U32, 2:U32)U32})U32}, Tensor_15_stride_1_14:I64)I64)U64}, 4:U32)U64, 0:U32)U64)U32
 // 	x = Convert({Join: AddShiftL(Convert({Flatten: ShiftLAdd({Workgroup Index X: s2:U32}, 6:U32, {Flatten: ShiftL({Tile[1]: Subtract({Workitem Index X: v1:U32}, BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32)U32}, 2:U32)U32})U32})U64, Multiply({Flatten: ShiftLAdd({Workgroup Index Y: s3:U32}, 6:U32, {Flatten: ShiftL(LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32, 2:U32)U32})U32}, Tensor_15_stride_1_14:I64)I64, 2:U32)U64})U32

 // (op 629) Get arg Tensor_15_stride_1_14
 // (op 629) reg expression
 // (op 629) reg expression
 // (op 629) reg expression
 // (op 629) reg expression
 // (op 629) BitwiseAnd({Workitem Index X: v1:U32}, -16:I)U32
 // (op 629) LogicalShiftR({Workitem Index X: v1:U32}, 4:I)U32
// Allocated : 1 VGPR (Value: UInt32) (op 629): v7
v_and_b32 v7, -16, v1 // (op 629) 
// Allocated : 1 VGPR (Value: UInt32) (op 629): v6
v_lshrrev_b32 v6, 4, v1 // (op 629) 
 // (op 629) BEGIN: Tile[1]
 // (op 629) {Tile[1]: Subtract({Workitem Index X: v1:U32}, v7:U32)U32}
 // (op 629) BEGIN: Flatten
 // (op 629) {Flatten: ShiftL(v6:U32, 2:U32)U32}
 // (op 629) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 629)         value (VGPR Value: UInt32 x 1: v6) 
 // (op 629)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 1 VGPR (Value: UInt32) (op 629): v24
v_sub_u32 v24, v1, v7 // (op 629) 
 // (op 629) END: Tile[1]
 // Freeing : 1 VGPR (Value: UInt32) (op 629): v7

// Allocated : 1 VGPR (Value: UInt32) (op 629): v7
v_lshlrev_b32 v7, 2, v6 // (op 629) 
 // (op 629) END: Flatten
 // Freeing : 1 VGPR (Value: UInt32) (op 629): v6

 // (op 629) BEGIN: Flatten
 // (op 629) {Flatten: ShiftL({Tile[1]: v24:U32}, 2:U32)U32}
 // (op 629) ShiftL: dest (VGPR Value: UInt32 x 1: (unallocated)) = 
 // (op 629)         value (Tile[1]: VGPR Value: UInt32 x 1: v24) 
 // (op 629)         shiftAmount (Literal Value: UInt32 x 0: 2)
 // (op 629) BEGIN: Flatten
// Allocated : 1 VGPR (Value: UInt32) (op 629): v6
v_lshlrev_b32 v6, 2, v24 // (op 629) 
 // (op 629) END: Flatten
 // Freeing Tile[1]: 1 VGPR (Value: UInt32) (op 629): v24

// Allocated : 1 VGPR (Value: UInt32) (op 629): v24
v_lshl_add_u32 v24, s3, 6, v7 // (op 629) 
 // (op 629) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 629): v7

 // (op 629) BEGIN: Flatten
 // (op 629) Multiply({Flatten: v24:U32}, {Tensor_15_stride_1_14: s[30:31]:I64})I64
 // (op 629) Multiply: dest (VGPR Value: Int64 x 1: (unallocated)) = 
 // (op 629)           lhs (Flatten: VGPR Value: UInt32 x 1: v24) 
 // (op 629)           rhs (Tensor_15_stride_1_14: SGPR Value: Int64 x 1: s[30:31])
// Allocated : 1 VGPR (Value: UInt32) (op 629): v7
v_lshl_add_u32 v7, s2, 6, v6 // (op 629) 
 // (op 629) END: Flatten
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 629): v6

// Allocated : 2 VGPRs (Value: Int64) (op 629): v26, v27
v_mul_lo_u32 v27, v24, s31 // (op 629) most significant: low of low * high
 // (op 629) low of high * low omitted due to zero input.
// Allocated : 1 VGPR (Value: Int32) (op 629): v6
v_mul_hi_u32 v6, v24, s30 // (op 629) most significant: high of low * low
v_mul_lo_u32 v26, v24, s30 // (op 629) least significant: low of low * low
v_add_u32 v27, v27, v6 // (op 629) most significant: sum
 // Freeing : 1 VGPR (Value: Int32) (op 629): v6

 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 629): v24

// Allocated Flatten: 2 VGPRs (Value: UInt64) (op 629): v24, v25
v_mov_b32 v25, 0 // (op 629) convert
v_mov_b32 v24, v7 // (op 629) convert
 // Freeing Flatten: 1 VGPR (Value: UInt32) (op 629): v7

 // (op 629) BEGIN: Join
 // (op 629) Add: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 629)      lhs (Flatten: VGPR Value: UInt64 x 1: v[24:25]) 
 // (op 629)      rhs (VGPR Value: Int64 x 1: v[26:27])
// Allocated : 2 VGPRs (Value: UInt64) (op 629): v6, v7
// Allocated : 2 SGPRs (Value: Bool64) (op 629): s0, s1
v_add_co_u32 v6, s[0:1], v24, v26 // (op 629) least significant half
v_addc_co_u32 v7, s[0:1], v25, v27, s[0:1] // (op 629) most significant half
 // Freeing : 2 SGPRs (Value: Bool64) (op 629): s0, s1

 // (op 629) ShiftL: dest (VGPR Value: UInt64 x 1: v[6:7]) = 
 // (op 629)         value (VGPR Value: UInt64 x 1: v[6:7]) 
 // (op 629)         shiftAmount (Literal Value: UInt32 x 0: 2)
v_lshlrev_b64 v[6:7], 2, v[6:7] // (op 629) 
 // (op 629) END: Join
 // Freeing : 2 VGPRs (Value: Int64) (op 629): v26, v27

 // Freeing Flatten: 2 VGPRs (Value: UInt64) (op 629): v24, v25

// Allocated Offset629: 1 VGPR (Value: UInt32) (op 629): v24
v_mov_b32 v24, v6 // (op 629) convert
 // Freeing Join: 2 VGPRs (Value: UInt64) (op 629): v6, v7

 // FastArithmetic:	orig = Add(Multiply(0:U32, 1:U64)U64, Multiply(1:U32, Tensor_15_stride_1_14:I64)I64)U64
 // 	x = Convert(Tensor_15_stride_1_14:I64)U64

 // (op 629)   Stride(496): indexExpr: Convert(Tensor_15_stride_1_14:I64)U64
 // (op 629)   Stride(496): indexExprPaddingBytes: 0:U32
 // (op 629)   Stride(496): unitStride: false vgprBlockSize: 0
 // (op 629)   Stride(496): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 629)   Stride(496): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(Convert(Tensor_15_stride_1_14:I64)U64, 4:U32)U64, 0:U32)U64
 // 	x = ShiftL(Convert(Tensor_15_stride_1_14:I64)U64, 2:U32)U64

 // tag 492: s**UNALLOCATED**
 // (op 629) Get arg Tensor_15_pointer
// Allocated Buffer492: 4 SGPRs (Buffer: None) (op 629): s4, s5, s6, s7
s_mov_b32 s4, s34 // (op 629) 
s_mov_b32 s5, s35 // (op 629) 
s_mov_b32 s7, 131072 // (op 629) default options
 // (op 629) Generate Multiply(Tensor_15_extent_15:I64, 4:U32)I64 into nullptr
 // FastArithmetic:	orig = Multiply(Tensor_15_extent_15:I64, 4:U32)I64
 // 	x = ShiftL(Tensor_15_extent_15:I64, 2:U32)I64

 // (op 629) Get arg Tensor_15_extent_15
 // (op 629) reg expression
 // (op 629) ShiftL({Tensor_15_extent_15: s[32:33]:I64}, 2:U32)I64
 // (op 629) ShiftL: dest (SGPR Value: Int64 x 1: (unallocated)) = 
 // (op 629)         value (Tensor_15_extent_15: SGPR Value: Int64 x 1: s[32:33]) 
 // (op 629)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 SGPRs (Value: Int64) (op 629): s0, s1
s_lshl_b64 s[0:1], s[32:33], 2 // (op 629) 
s_mov_b32 s6, s0 // (op 629) 
 // Freeing : 2 SGPRs (Value: Int64) (op 629): s0, s1

 // (op 528)  Tag 629non referenced 	extraArgs = {Tensor_0_size_0_8, Tensor_2_size_1_12}

 // (op 528) ComputeIndex(629) END
 // (op 528) ComputeIndex(630) BEGIN
 // (op 630) KernelGraph::LoadStoreTileGenerator::ComputeIndex(630): target 47 increment 270 base 495 offset 497 stride 498 buffer -1
 // FastArithmetic:	orig = Add(Multiply(1:U32, 1:U64)U64, Multiply(0:U32, Tensor_15_stride_1_14:I64)I64)U64
 // 	x = 1:U64

 // (op 630)   Stride(498): indexExpr: 1:U64
 // (op 630)   Stride(498): indexExprPaddingBytes: 0:U32
 // (op 630)   Stride(498): unitStride: true vgprBlockSize: 0
 // (op 630)   Stride(498): elementBlockStride: nullptr elementBlockStridePaddingBytes: 0:U32
 // (op 630)   Stride(498): trLoadPairStride:  nullptr trLoadPairStridePaddingBytes: 0:U32
 // FastArithmetic:	orig = Add(Multiply(1:U64, 4:U32)U64, 0:U32)U64
 // 	x = 4:U64

 // (op 528)  Tag 630non referenced 	extraArgs = {Tensor_0_size_0_8, Tensor_15_extent_15, Tensor_15_stride_1_14, Tensor_2_size_1_12}

 // (op 528) ComputeIndex(630) END
 // (op 528) StoreTiled Value: Float(480) BEGIN
 // (op 480) GEN: storeMacroTileVGPR OP 480 MacroTile 429
 // (op 480) 	varTypeInfo.elementBits = 32
 // 	varTypeInfo.packing = 1

 // (op 480) Generate ShiftL(Convert(Tensor_15_stride_1_14:I64)U64, 2:U32)U64 into VGPR Value: UInt64 x 1: (unallocated)
 // (op 480) Get arg Tensor_15_stride_1_14
 // (op 480) reg expression
 // (op 480) convert
 // (op 480) ShiftL({ convertInPlaceTensor_15_stride_1_14: s[30:31]:U64}, 2:U32)U64
 // (op 480) ShiftL: dest (VGPR Value: UInt64 x 1: (unallocated)) = 
 // (op 480)         value ( convertInPlaceTensor_15_stride_1_14: SGPR Value: UInt64 x 1: s[30:31]) 
 // (op 480)         shiftAmount (Literal Value: UInt32 x 0: 2)
// Allocated : 2 VGPRs (Value: UInt64) (op 480): v6, v7
v_lshlrev_b64 v[6:7], 2, s[30:31] // (op 480) 
 // (op 480) Generate 4:U64 into nullptr
 // (op 480) 	Dir = Store
 // LSTInfo {
 // 	info.kind = Buffer
 // 	info.m = 4
 // 	info.n = 4
 // 	info.elementBits = 32
 // 	info.packedAmount = 1
 // 	info.ldsWriteStride = 0
 // 	info.data = VGPR Value: Float x 16: v[8:23]
 // 	info.rowOffsetReg = (nullptr)
 // 	info.rowStrideReg = VGPR Value: UInt64 x 1: v[6:7]
 // 	info.rowStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 0
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.colStrideReg = Literal Value: UInt64 x 0: 4
 // 	info.colStrideAttributes = {
 // 	t.dataType = UInt64
 // 	t.unitStride = 1
 // 	t.elementBlockSize = 0
 // 	t.elementBlockStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // 	t.trLoadPairStride = Add(Multiply(nullptr, 4:U32)NA, 0:U32)NA
 // }
 // 	info.offset = Literal Value: UInt32 x 0: 0
 // 	info.bufDesc = Buffer492: SGPR Buffer: None x 1: s[4:7]
 // 	info.bufOpts = {	options.offen = 0
 // 	options.glc = 0
 // 	options.slc = 0
 // 	options.sc1 = 0
 // 	options.lds = 0
 // }
 // 	info.isTransposedTile = 0
 // }	allStridesAreLiteral = 0
 // 	colStrideIsOne = 1

// Allocated : 1 VGPR (Value: UInt32) (op 480): v25
v_mov_b32 v25, v24 // (op 480) 
 // (op 480)   M 4 N 4 elementsPerMove 4 bytesPerMove 16 rowStride v[6:7]:U64 colStride 4:U64 vgprBlockSize 0 numVGPRBlocks 1
s_waitcnt lgkmcnt(3)
buffer_store_dwordx4 v[8:11], v25, s[4:7], 0 offen // (op 480) Store value
// VMEM: Expected complete at 1286 (current 1211)
 // (op 480) Generate Add(v25:U32, v6:R)U32 into VGPR Value: UInt32 x 1: v25
 // (op 480) reg expression
 // (op 480) reg expression
 // (op 480) Add(v25:U32, v6:R)U32
s_nop 1
v_add_u32 v25, v25, v6 // (op 480) Wait state hazard: Buffer Store Read Hazard
s_waitcnt lgkmcnt(2)
buffer_store_dwordx4 v[12:15], v25, s[4:7], 0 offen // (op 480) Store value
// VMEM: Expected complete at 1291 (current 1216)
 // (op 480) Generate Add(v25:U32, v6:R)U32 into VGPR Value: UInt32 x 1: v25
 // (op 480) reg expression
 // (op 480) reg expression
 // (op 480) Add(v25:U32, v6:R)U32
s_nop 1
v_add_u32 v25, v25, v6 // (op 480) Wait state hazard: Buffer Store Read Hazard
s_waitcnt lgkmcnt(1)
buffer_store_dwordx4 v[16:19], v25, s[4:7], 0 offen // (op 480) Store value
// VMEM: Expected complete at 1296 (current 1221)
 // (op 480) Generate Add(v25:U32, v6:R)U32 into VGPR Value: UInt32 x 1: v25
 // (op 480) reg expression
 // (op 480) reg expression
 // (op 480) Add(v25:U32, v6:R)U32
s_nop 1
v_add_u32 v25, v25, v6 // (op 480) Wait state hazard: Buffer Store Read Hazard
s_waitcnt lgkmcnt(0)
buffer_store_dwordx4 v[20:23], v25, s[4:7], 0 offen // (op 480) Store value
// VMEM: Expected complete at 1301 (current 1226)
 // Freeing : 2 VGPRs (Value: UInt64) (op 480): v6, v7

 // Freeing : 1 VGPR (Value: UInt32) (op 480): v25

 // (op 528) StoreTiled Value: Float(480) END
 // (op 528) Deallocate{}(728) BEGIN
 // (op 728) Deallocate 429
 // Freeing : 16 VGPRs (Value: Float) (op 479): v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23

 // Deleting tag 429
 // (op 528) Deallocate{}(728) END
 // (op 528) end: generate({531})
 // Freeing Buffer492: 4 SGPRs (Buffer: None) (op 629): s4, s5, s6, s7

 // Deleting tag 492
 // Freeing Offset629: 1 VGPR (Value: UInt32) (op 629): v24

 // Deleting tag 495
 // (op 528) Unlock Scope 528
 // (op 467) Scope(528) END
 // (op 467) Deallocate{Tensor_15_extent_15, Tensor_15_pointer, Tensor_15_stride_1_14}(796) BEGIN
 // (op 796) Deallocate Tensor_15_extent_15
 // Freeing Tensor_15_extent_15: 2 SGPRs (Value: Raw32): s32, s33

 // (op 796) Deallocate Tensor_15_pointer
 // Freeing Tensor_15_pointer: 2 SGPRs (Value: Raw32): s34, s35

 // (op 796) Deallocate Tensor_15_stride_1_14
 // Freeing Tensor_15_stride_1_14: 2 SGPRs (Value: Raw32): s30, s31

 // (op 467) Deallocate{Tensor_15_extent_15, Tensor_15_pointer, Tensor_15_stride_1_14}(796) END
 // (op 467) end: generate({528})
GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_ConditionalBottom_0_beta_467:
 // (op 467) 
 // (op 467) Unlock Conditional
 // (op 1) ConditionalOp 0 == beta: Equal(0.00000:S, DataFlowTag(25)S)BL(467) END
 // (op 1) Deallocate{Tensor_2_size_1_12, Tensor_0_size_0_8}(724) BEGIN
 // (op 724) Deallocate 36
 // Freeing DataFlowTag36: 16 ACCVGPRs (Value: Float) (op 46): a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15

 // Deleting tag 36
 // (op 724) Deallocate 34
 // Freeing : 1 VGPR (Value: Float) (op 24): v5

 // Deleting tag 34
 // (op 724) Deallocate 25
 // Freeing : 1 VGPR (Value: Float) (op 2): v3

 // Deleting tag 25
 // (op 724) Deallocate Tensor_2_size_1_12
 // Freeing Tensor_2_size_1_12: 2 SGPRs (Value: Raw32): s26, s27

 // (op 724) Deallocate Tensor_0_size_0_8
 // Freeing Tensor_0_size_0_8: 2 SGPRs (Value: Raw32): s18, s19

 // (op 1) Deallocate{Tensor_2_size_1_12, Tensor_0_size_0_8}(724) END
 // (op 1) end: generate({593, 602})
 // Freeing Offset602: 1 VGPR (Value: UInt32) (op 602): v4

 // Deleting tag 477
 // Freeing Offset593: 1 VGPR (Value: UInt32) (op 593): v0

 // Deleting tag 469
 // Kernel(1) END
 // end: generate({1})
 // CodeGeneratorVisitor::generate() end
s_endpgm  // End of GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel
.LGEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_end:

.size GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel, .LGEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel_end-GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel
.rodata
.p2align 6
.amdhsa_kernel GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel
 // Resource limits
  .amdhsa_next_free_vgpr 72
  .amdhsa_next_free_sgpr .amdgcn.next_free_sgpr
  .amdhsa_group_segment_fixed_size 32768 // lds bytes
  .amdhsa_accum_offset 56
 // Initial kernel state
  .amdhsa_user_sgpr_kernarg_segment_ptr 1
.amdhsa_system_sgpr_workgroup_id_x 1
.amdhsa_system_sgpr_workgroup_id_y 1
.amdhsa_system_sgpr_workgroup_id_z 0
.amdhsa_system_sgpr_workgroup_info 0
.amdhsa_system_vgpr_workitem_id 1
.end_amdhsa_kernel

.amdgpu_metadata
---
amdhsa.version: [1, 2]
amdhsa.kernels:
  - .name: GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel
    .symbol: GEMMTest_GEMMTestGPUGPU_BasicGEMM_0_kernel.kd
    .kernarg_segment_size: 128
    .group_segment_fixed_size: 32768
    .private_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .sgpr_count: 44
    .vgpr_count: 56
    .agpr_count: 16
    .max_flat_workgroup_size: 256
    .workgroup_size:
      - 256
      - 1
      - 1
    .kernel_dimensions: 2
    .wavefront_size: 64
    .workitem_count: [{type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_0_size_0_8, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 56, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 256}}, {type: Multiply, lhs: {type: Convert, arg: {type: ArithmeticShiftR, lhs: {type: Subtract, lhs: {type: Add, lhs: {type: Kernel Argument, name: Tensor_2_size_1_12, variableType: {dataType: Int64, pointerType: Value}, dataDirection: read_only, expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}, offset: 88, size: 8}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 64}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 6}}, dataType: Int32}, rhs: {type: LiteralValue.UInt32, dataType: UInt32, value: 1}}, {is-null: true}]
    .dynamic_sharedmemory_bytes: {type: LiteralValue.UInt32, dataType: UInt32, value: 0}
    .args:
      - .name: Tensor_0_extent_0
        .size: 8
        .offset: 0
        .expression: {type: CommandArgument, size: 8, offset: 8, name: Tensor_0_extent, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_pointer
        .size: 8
        .offset: 8
        .expression: {type: CommandArgument, size: 8, offset: 0, name: Tensor_0_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_2_extent_2
        .size: 8
        .offset: 16
        .expression: {type: CommandArgument, size: 8, offset: 56, name: Tensor_2_extent, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_pointer
        .size: 8
        .offset: 24
        .expression: {type: CommandArgument, size: 8, offset: 48, name: Tensor_2_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: Tensor_4_extent_4
        .size: 8
        .offset: 32
        .expression: {type: CommandArgument, size: 8, offset: 104, name: Tensor_4_extent, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_pointer
        .size: 8
        .offset: 40
        .expression: {type: CommandArgument, size: 8, offset: 96, name: Tensor_4_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
      - .name: user_Float_Value_6
        .size: 4
        .offset: 48
        .expression: {type: CommandArgument, size: 4, offset: 144, name: user_Float_Value_6, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: user_Float_Value_8
        .size: 4
        .offset: 52
        .expression: {type: CommandArgument, size: 4, offset: 148, name: user_Float_Value_8, variableType: {dataType: Float, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Float, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_0_8
        .size: 8
        .offset: 56
        .expression: {type: CommandArgument, size: 8, offset: 16, name: Tensor_0_size_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_size_1_9
        .size: 8
        .offset: 64
        .expression: {type: CommandArgument, size: 8, offset: 24, name: Tensor_0_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_0_stride_1_10
        .size: 8
        .offset: 72
        .expression: {type: CommandArgument, size: 8, offset: 40, name: Tensor_0_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_stride_0_11
        .size: 8
        .offset: 80
        .expression: {type: CommandArgument, size: 8, offset: 80, name: Tensor_2_stride_0, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_2_size_1_12
        .size: 8
        .offset: 88
        .expression: {type: CommandArgument, size: 8, offset: 72, name: Tensor_2_size_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_4_stride_1_13
        .size: 8
        .offset: 96
        .expression: {type: CommandArgument, size: 8, offset: 136, name: Tensor_4_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_15_stride_1_14
        .size: 8
        .offset: 104
        .expression: {type: CommandArgument, size: 8, offset: 192, name: Tensor_15_stride_1, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_15_extent_15
        .size: 8
        .offset: 112
        .expression: {type: CommandArgument, size: 8, offset: 160, name: Tensor_15_extent, variableType: {dataType: Int64, pointerType: Value}, direction: read_only}
        .variableType: {dataType: Int64, pointerType: Value}
        .value_kind: by_value
      - .name: Tensor_15_pointer
        .size: 8
        .offset: 120
        .expression: {type: CommandArgument, size: 8, offset: 152, name: Tensor_15_pointer, variableType: {dataType: Float, pointerType: PointerGlobal}, direction: read_write}
        .variableType: {dataType: Float, pointerType: PointerGlobal}
        .address_space: global
        .actual_access: read_write
        .value_kind: global_buffer
...

.end_amdgpu_metadata