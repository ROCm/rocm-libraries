/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <string>

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/Arithmetic/ArithmeticGenerator.hpp>
#include <rocRoller/CodeGen/Arithmetic/MatrixMultiply.hpp>
#include <rocRoller/CodeGen/BranchGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/InstructionValues/LabelAllocator.hpp>
#include <rocRoller/InstructionValues/Register.hpp>

#include "GemmGuidePostKernels.hpp"

using namespace rocRoller;

Generator<Instruction> rocRollerTest::SGEMM_Minimal_Program(rocRoller::ContextPtr m_context)
{
    auto fmm = Component::Get<rocRoller::InstructionGenerators::MatrixMultiply>(m_context);

    auto label_4 = m_context->labelAllocator()->label("label_0013");
    auto label_2 = m_context->labelAllocator()->label("openLoopL_12");
    auto label_0 = m_context->labelAllocator()->label("ShadowInitStart_10");
    auto label_3 = m_context->labelAllocator()->label("LoopBeginL_1");
    auto label_5 = m_context->labelAllocator()->label("LoopEndL_2");
    auto label_7 = m_context->labelAllocator()->label("GW_B0_E0_17");
    auto label_1 = m_context->labelAllocator()->label("label_NoBranch_11");
    auto label_6 = m_context->labelAllocator()->label("Summation_End_14");

    auto vgprLocalWriteAddrA
        = m_context->kernel()->workitemIndex()[1]; // Explicitly use this vgpr we never use.
    auto vgprLocalWriteAddrB
        = m_context->kernel()->workitemIndex()[2]; // Explicitly use this vgpr we never use.
    auto vgprGlobalReadOffsetA
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    auto vgprGlobalReadOffsetB
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    auto vgprG2LA
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    auto vgprG2LB
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    auto vgprLocalReadAddrA
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    auto vgprLocalReadAddrB
        = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    auto vgprSerial = m_context->kernel()->workitemIndex()[0];

    auto accDestination
        = std::make_shared<Register::Value>(m_context,
                                            Register::Type::Accumulator,
                                            DataType::Float,
                                            16,
                                            Register::AllocationOptions::FullyContiguous());

    auto sgprWorkGroup0 = m_context->kernel()->workgroupIndex()[0];
    auto sgprWorkGroup1 = m_context->kernel()->workgroupIndex()[1];
    auto sgprWorkGroup2 = m_context->kernel()->workgroupIndex()[2];

    //Register::ValuePtr sizeC;
    //co_yield m_context->argLoader()->getValue("sizeC", sizeC);
    Register::ValuePtr sizeA;
    co_yield m_context->argLoader()->getValue("sizeA", sizeA);
    Register::ValuePtr sizeB;
    co_yield m_context->argLoader()->getValue("sizeB", sizeB);
    Register::ValuePtr D;
    co_yield m_context->argLoader()->getValue("D", D);
    Register::ValuePtr C;
    co_yield m_context->argLoader()->getValue("C", C);
    Register::ValuePtr A;
    co_yield m_context->argLoader()->getValue("A", A);
    Register::ValuePtr B;
    co_yield m_context->argLoader()->getValue("B", B);
    Register::ValuePtr sgprStrideD1J;
    co_yield m_context->argLoader()->getValue("strideD0", sgprStrideD1J);
    Register::ValuePtr sgprStrideDK;
    co_yield m_context->argLoader()->getValue("strideD1", sgprStrideDK);
    Register::ValuePtr sgprStrideC1J;
    co_yield m_context->argLoader()->getValue("strideC0", sgprStrideC1J);
    Register::ValuePtr sgprStrideCK;
    co_yield m_context->argLoader()->getValue("strideC1", sgprStrideCK);
    Register::ValuePtr sgprStrideAL;
    co_yield m_context->argLoader()->getValue("strideA0", sgprStrideAL);
    Register::ValuePtr sgprStrideAK;
    co_yield m_context->argLoader()->getValue("strideA1", sgprStrideAK);
    Register::ValuePtr sgprStrideBL;
    co_yield m_context->argLoader()->getValue("strideB0", sgprStrideBL);
    Register::ValuePtr sgprStrideBK;
    co_yield m_context->argLoader()->getValue("strideB1", sgprStrideBK);
    Register::ValuePtr sgprSizesSum;
    co_yield m_context->argLoader()->getValue("SizesSum0", sgprSizesSum);
    Register::ValuePtr sgprGlobalReadIncsB;
    co_yield m_context->argLoader()->getValue("OrigStaggerUIter", sgprGlobalReadIncsB);
    Register::ValuePtr OffsetD;
    co_yield m_context->argLoader()->getValue("OffsetD", OffsetD);
    Register::ValuePtr OffsetC;
    co_yield m_context->argLoader()->getValue("OffsetC", OffsetC);
    Register::ValuePtr OffsetA;
    co_yield m_context->argLoader()->getValue("OffsetA", OffsetA);
    Register::ValuePtr OffsetB;
    co_yield m_context->argLoader()->getValue("OffsetB", OffsetB);
    //Register::ValuePtr padding;
    //co_yield m_context->argLoader()->getValue("padding", padding);
    m_context->argLoader()->releaseAllArguments();

    auto sgprShadowLimitA
        = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Int64, 1);
    auto sgprShadowLimitB
        = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Int64, 1);
    auto sgprLoopCounterL
        = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Int32, 1);
    auto sgprGlobalReadIncsA
        = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Int32, 1);

    VariableType bufferPointer{DataType::None, PointerType::Buffer};
    auto         sgprSrdA
        = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, bufferPointer, 1);
    co_yield sgprSrdA->allocate();
    auto sgprSrdB
        = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, bufferPointer, 1);
    co_yield sgprSrdB->allocate();
    auto sgprSrdD
        = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, bufferPointer, 1);
    co_yield sgprSrdD->allocate();
    auto sgprSrdC
        = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, bufferPointer, 1);
    co_yield sgprSrdC->allocate();

    auto macroTile0 = Register::Value::Literal(64);
    auto macroTile1 = Register::Value::Literal(64);
    int  DepthU = 4, BpeA = 4, BpeB = 4;

    // clang-format off
co_yield Instruction::Comment("****************************************"
                              " Begin Kernel                           \n"
                              "****************************************\n"
                              " Component.Signature.SignatureCOV3\n"
                              "****************************************\n"
                              " Optimizations and Config:              \n"
                              "****************************************\n"
                              " ThreadTile= 16 x 1 \n"
                              " SubGroup= 4 x 64 \n"
                              " VectorWidth=1 \n"
                              " GlobalLoadVectorWidthA=1, GlobalLoadVectorWidthB=1 \n"
                              " DirectToLdsA=False \n"
                              " DirectToLdsB=False \n"
                              " UseSgprForGRO=1 \n"
                              " Number of elements to shift-left SRD \n"
                              " 2GB limit - set offsets to -1 to exceed this and clamp \n");
auto BufferLimit = Register::Value::Label("0xffffffff");
co_yield Instruction::Comment("****************************************\n"
                              " Bits 127:96 of SRD.                    \n"
                              " hex: 0x00020000                        \n"
                              " dst_sel_x (3b): 0                      \n"
                              " dst_sel_y (3b): 0                      \n"
                              " dst_sel_z (3b): 0                      \n"
                              " dst_sel_w (3b): 0                      \n"
                              " num_format (3b): 0                     \n"
                              " data_format (4b): 4                    \n"
                              " user_vm_enable (1b): 0                 \n"
                              " user_vm_mode (1b): 0                   \n"
                              " index_stride (2b): 0                   \n"
                              " add_tid_enable (1b): 0                 \n"
                              " _unusedA (3b): 0                       \n"
                              " nv (1b): 0                             \n"
                              " _unusedB (2b): 0                       \n"
                              " type (2b): 0                           \n"
                              "****************************************\n");
auto Srd127_96 = Register::Value::Literal(131072); //0x00020000
co_yield Instruction::Comment("****************************************\n"
                              " Allocate Resources                     \n"
                              "****************************************\n");
co_yield_(Instruction("s_mov_b32", {Register::Value::Label("m0")}, {Register::Value::Literal(4096)}, {}, " LDS clamp at 4096 bytes"));
//co_yield Instruction::Wait(WaitCount::LGKMCnt(0, "wait for 148 bytes of kern args"));
co_yield generateOp<Expression::ShiftL>(OffsetD, OffsetD, Register::Value::Literal(2));
co_yield generateOp<Expression::Add>(D, D, OffsetD);
co_yield generateOp<Expression::ShiftL>(OffsetC, OffsetC, Register::Value::Literal(2));
co_yield generateOp<Expression::Add>(C, C, OffsetC);
co_yield generateOp<Expression::ShiftL>(OffsetA, OffsetA, Register::Value::Literal(2));
co_yield generateOp<Expression::Add>(A, A, OffsetA);
co_yield generateOp<Expression::ShiftL>(OffsetB, OffsetB, Register::Value::Literal(2));
co_yield generateOp<Expression::Add>(B, B, OffsetB);
co_yield generateOp<Expression::Subtract>(A, A, Register::Value::Literal(4)); //pre-pad to make room for possible pointer shift
co_yield generateOp<Expression::Subtract>(B, B, Register::Value::Literal(4)); //pre-pad to make room for possible pointer shift
co_yield Instruction::Comment("****************************************\n"
                              " Local Read Addresses                   \n"
                              "****************************************\n");

auto LRAddresses = [&](Register::ValuePtr     macroTile,
                       Register::ValuePtr     vgprLocalReadAddr,
                       int                    dividedForWaveId) -> Generator<Instruction>
{
    auto stemp = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Int32, 1);
    auto vtemp = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    auto vtemp2 = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    auto vtemp3 = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    auto lro = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    co_yield Instruction::Comment(" local read addresses: tile assignments");
    co_yield generateOp<Expression::BitwiseAnd>(vtemp3, Register::Value::Literal(63), vgprSerial); //" 0. thread id in wave: wtid = tid % wavelength(64)"
    co_yield generateOp<Expression::BitwiseAnd>(lro, Register::Value::Literal(31), vtemp3); //" 1. N offset: nIdx = wtid % MI_N(32)"
    co_yield Instruction::Comment(" 1. N offset: nOffset = nIdx * nStride(1) (multiplier is 1, do nothing)");
    co_yield generateOp<Expression::LogicalShiftR>(vtemp2, vtemp3, Register::Value::Literal(5)); //" 2. block offset: bnIdx = wtid / dividedForBlkId(32)"
    co_yield generateOp<Expression::BitwiseAnd>(vtemp2, Register::Value::Literal(0), vtemp2); //" 2. block offset: bnIdx = bnIdx % num1DBlocks(1)"
    //" 2. block offset: bnOffset = bnIdx * strideBlock(32)"
    //" 3. add N and block offset: bnOffset = block and N offset"
    co_yield generateOp<Expression::ShiftLAdd>(lro, vtemp2, Register::Value::Literal(5), lro);
    co_yield Instruction::Comment(" 3. apply VectorWidth: bnOffset = bnOffset * vw(1) (multiplier is 1, do nothing)");
    co_yield generateOp<Expression::LogicalShiftR>(vtemp3, vtemp3, Register::Value::Literal(5)); //" 4. K offset: kIdx = wtid / (MIN(32) * MIBB(1))"
    //" 4. K offset: lrKOffset = kIdx * mStride(64)"
    //" 5. offset in wave: lrOffset = bnOffset + lrKOffset"
    co_yield generateOp<Expression::ShiftLAdd>(lro, vtemp3, Register::Value::Literal(6), lro);
    co_yield generateOp<Expression::LogicalShiftR>(vtemp2, vgprSerial, Register::Value::Literal(dividedForWaveId)); //" 6. wave offset in N dimen: wtid = tid / dividedForWaveId(64)"
    co_yield generateOp<Expression::BitwiseAnd>(vtemp2, Register::Value::Literal(1), vtemp2); //" 6. wave offset in M dimen: wtid0 = wtid / num1DWaves(2)"
    //" 6. wave offset in M dimen: wOffset = wtid0 * W0Stride(32)"
    //" 7. final local read offset: flrOffset = lrOffset + WOffset"
    co_yield generateOp<Expression::ShiftLAdd>(lro, vtemp2, Register::Value::Literal(5), lro);
    co_yield Instruction::Comment(" local read addresses: final offsets");
    co_yield generateOp<Expression::LogicalShiftR>(vtemp, vgprSerial, Register::Value::Literal(8)); //" LSU offset: sgid = Serial / subGroup(256)"
    co_yield m_context->copier()->copy(stemp, macroTile, " LSU offset: stride = MT(64) + PAD(0)");
    co_yield generateOp<Expression::Multiply>(vtemp, stemp, vtemp); //" LSU offset: lsuoffset = sgid*(MT+PAD)"
    co_yield_(Instruction("v_add_lshl_u32", {vgprLocalReadAddr}, {vtemp, lro, Register::Value::Literal(2)}, {}, " Final Offset: offset = (lro*VW+lsuoffset)*bpe"));
};
co_yield Instruction::Comment(" local read addresses: a ");
co_yield LRAddresses(macroTile0, vgprLocalReadAddrA, 6);
co_yield Instruction::Comment(" local read addresses: b ");
co_yield LRAddresses(macroTile1, vgprLocalReadAddrB, 7);
co_yield Instruction::Comment(" local read addresses: declare addresses a \n"
                              " N/A \n"
                              " local read addresses: declare addresses b \n");
co_yield_(Instruction("v_add_co_u32", {vgprLocalReadAddrB}, {m_context->getVCC(), Register::Value::Literal(1024), vgprLocalReadAddrB}, {}, "  += LdsOffsetB (lower)"));
co_yield Instruction::Comment("****************************************\n"
                              " Begin setupNewTile, isPap=False           \n"
                              "****************************************\n"
                              " global read addresses: work-group \n"
                              " graWorkGroup mapping \n");
co_yield Instruction::Comment(" global read addresses: unroll assignment a \n"
                              " v1 \n"
                              " global read addresses: unroll assignment b \n"
                              " v3 \n"
                              " global read addresses: other free assignments \n"
                              " s[sgprWorkGroup2] \n"
                              " global read addresses: tile offsets a \n"
                              " global read addresses: tile offsets b \n"
                              " global read addresses: unroll offsets a \n"
                              " global read addresses: unroll offsets b \n");
auto GlobalOffset = [&](Register::ValuePtr     vgprAddr,
                        Register::ValuePtr     sgpr,
                        Register::ValuePtr     vgprOffset,
                        Register::ValuePtr     vgprOffsetL) -> Generator<Instruction>
{
    auto vgprTmp = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
    co_yield Instruction::Comment(" global read addresses: tile offset assignment\n"
                                " LVCB = 64 \n"
                                " v2 = (local)groB-tile = serial%LVCB (note (wgB*MTB) will be added to SRD) \n"
                                " v3 = groB-unroll = serial/LVCB \n");
    co_yield generateOp<Expression::LogicalShiftR>(vgprOffsetL, vgprSerial, Register::Value::Literal(6)); //" vgprOffsetL = v[vgprSerial] / 64"
    co_yield generateOp<Expression::BitwiseAnd>(vgprOffset, Register::Value::Literal(63), vgprSerial); //" vgprOffset = v[vgprSerial] % 64"
    co_yield Instruction::Comment(" global read addresses: final offsets");
    co_yield generateOp<Expression::Multiply>(vgprTmp, sgpr, vgprOffsetL); //"mul d1 lower"
    co_yield_(Instruction("v_add_co_u32", {vgprAddr}, {m_context->getVCC(), vgprOffset, vgprTmp}, {}, "accumulate K lower"));
    //"add prepad for pointer shift"
    //"offset *= bytes/element"
    co_yield generateOp<Expression::AddShiftL>(vgprAddr, Register::Value::Literal(1), vgprAddr, Register::Value::Literal(2));
};
auto GlobalReadAddresses = [&](Register::ValuePtr     val,
                               Register::ValuePtr     size,
                               Register::ValuePtr     sgprWorkGroup,
                               Register::ValuePtr     sgprShadowLimit,
                               Register::ValuePtr     sgprSrd,
                               Register::ValuePtr     sgprStride) -> Generator<Instruction>
{
    auto tileStart = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Int64, 1);
    auto stemp = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Int64, 1);
    co_yield Instruction::Comment(" max read offset = size[n] * stride[n-1] ");
    co_yield generateOp<Expression::Multiply>(tileStart, sgprWorkGroup, macroTile0); //" WorkGroup[01] * MT"
    co_yield generateOp<Expression::Subtract>(sgprShadowLimit, size, tileStart); //" sub tileStart"
    co_yield generateOp<Expression::ShiftL>(sgprShadowLimit, sgprShadowLimit, Register::Value::Literal(2)); //" Set limit to use bytes"
    co_yield generateOp<Expression::Add>(sgprShadowLimit, sgprShadowLimit, Register::Value::Literal(4)); //" extend limit for pre-pad"
    co_yield generateOp<Expression::Equal>(nullptr, sgprShadowLimit->subset({1}), Register::Value::Literal(0)); //" are we within 2^32?"
    co_yield_(Instruction("s_cselect_b32", {sgprSrd->subset({2})}, {sgprShadowLimit->subset({0}), BufferLimit}, {}, " Move shadow to real if we are within 2^32"));
    co_yield generateOp<Expression::Multiply>(stemp, sgprStride, sgprWorkGroup2); //" Stride*WG"
    co_yield generateOp<Expression::Add>(tileStart, tileStart, stemp); //" accum wg term to tilestart"
    co_yield generateOp<Expression::ShiftL>(tileStart, tileStart, Register::Value::Literal(2)); //" tileStart *= BPE"
    co_yield generateOp<Expression::Add>(sgprSrd->subset({0,1}), val, tileStart); //" SRD base = Address+ tileStart0"
    co_yield m_context->copier()->copy(sgprSrd->subset({3}), Srd127_96, " Set bits 127_96 in SRD");
};
{
auto vgprOffsetA = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
auto vgprOffsetLA = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
auto vgprOffsetB = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
auto vgprOffsetLB = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
co_yield GlobalOffset(vgprGlobalReadOffsetA, sgprStrideAL, vgprOffsetA, vgprOffsetLA);
co_yield GlobalOffset(vgprGlobalReadOffsetB, sgprStrideBL, vgprOffsetB, vgprOffsetLB);
co_yield Instruction::Comment(" global read addresses: addresses a");
co_yield GlobalReadAddresses(A, sizeA, sgprWorkGroup0, sgprShadowLimitA, sgprSrdA, sgprStrideAK);
co_yield Instruction::Comment(" global read addresses: addresses b");
co_yield GlobalReadAddresses(B, sizeB, sgprWorkGroup1, sgprShadowLimitB, sgprSrdB, sgprStrideBK);
co_yield Instruction::Comment(" global read addresses: increments a ");
co_yield generateOp<Expression::Multiply>(sgprGlobalReadIncsA, Register::Value::Literal(DepthU*BpeA), sgprStrideAL); //" incrA unrollIdx)"
co_yield Instruction::Comment(" global read addresses: increments b ");
co_yield generateOp<Expression::Multiply>(sgprGlobalReadIncsB, Register::Value::Literal(DepthU*BpeB), sgprStrideBL); //" incrB unrollIdx)"
co_yield Instruction::Comment("****************************************\n"
                              " Local Write Addresses                  \n"
                              "****************************************\n"
                              " lwaTileAssignmentA = v0 \n"
                              " lwaTileAssignmentB = v2 \n"
                              " lwaUnrollAssignmentA = v1 \n"
                              " lwaUnrollAssignmentB = v3 \n"
                              " local write addresses: first offset a \n");
co_yield_(Instruction("v_mul_u32_u24", {vgprLocalWriteAddrA}, {macroTile0, vgprOffsetLA}, {}, " lwAL**(MTA + PAD)"));
co_yield_(Instruction("v_add_lshl_u32", {vgprLocalWriteAddrA}, {vgprOffsetA, vgprLocalWriteAddrA, Register::Value::Literal(2)}, {}, " lwFOA = (lwAA + lwAL*(MT0I+PAD))*bpe"));
co_yield Instruction::Comment(" local write addresses: first offset b ");
co_yield_(Instruction("v_mul_u32_u24", {vgprLocalWriteAddrB}, {macroTile1, vgprOffsetLB}, {}, " lwBL**(MTB + PAD)"));
co_yield_(Instruction("v_add_lshl_u32", {vgprLocalWriteAddrB}, {vgprOffsetB, vgprLocalWriteAddrB, Register::Value::Literal(2)}, {}, " lwFOB = (lwBB + lwBL*(MT1J+PAD))*bpe"));
//Could be add vvv
co_yield_(Instruction("v_add_co_u32", {vgprLocalWriteAddrB, m_context->getVCC()}, {Register::Value::Literal(1024), vgprLocalWriteAddrB}, {}, " lwFOB = lwB1J + lwBL*MT1J + LDS_OFFSET_B=256*4"));
}
co_yield Instruction::Comment(" declare loop num iterations ");
co_yield generateOp<Expression::LogicalShiftR>(sgprLoopCounterL, sgprSizesSum, Register::Value::Literal(2)); //" s[sgprLoopCounterL] = s[sgprSizesSum+0] / 4"
co_yield Instruction::Comment(" local read addresses: init pointers a \n"
                              " localReadInitPointers \n"
                              " local read addresses: init pointers b \n"
                              " localReadInitPointers \n"
                              " prefetch: global -> local \n");
co_yield generateOp<Expression::Equal>(nullptr, sgprLoopCounterL, Register::Value::Literal(0));// " at last iteration?"
co_yield m_context->brancher()->branchIfNonZero(label_0, m_context->getSCC(), " skip to ShadowInitStart iter b/c numIter==0");
co_yield_(Instruction("buffer_load_dword", {vgprG2LA}, {vgprGlobalReadOffsetA, sgprSrdA, Register::Value::Literal(0)}, {"offen", "offset:0"}, " G -> Reg 0_0_0_0"));
co_yield_(Instruction("buffer_load_dword", {vgprG2LB}, {vgprGlobalReadOffsetB, sgprSrdB, Register::Value::Literal(0)}, {"offen", "offset:0"}, " G -> Reg 0_0_0_0"));
auto GlobalReadInc = [&](Register::ValuePtr     sgprSrd,
                         Register::ValuePtr     sgprShadowLimit,
                         Register::ValuePtr     sgprGlobalReadIncs) -> Generator<Instruction>
{
    co_yield generateOp<Expression::Add>(sgprSrd->subset({0, 1}), sgprSrd->subset({0, 1}), sgprGlobalReadIncs); //" gra SRD += inc"
    co_yield generateOp<Expression::Subtract>(sgprShadowLimit, sgprShadowLimit, sgprGlobalReadIncs); //" limit -= inc"
    co_yield generateOp<Expression::Equal>(nullptr, sgprShadowLimit->subset({1}), Register::Value::Literal(0)); //" are we within 2^32?"
    co_yield m_context->copier()->conditionalCopy(sgprSrd->subset({2}), sgprShadowLimit->subset({0}), " Move shadow to real if we are within 2^32");
};
co_yield Instruction::Comment(" global read inc A loopL ");
co_yield GlobalReadInc(sgprSrdA, sgprShadowLimitA, sgprGlobalReadIncsA);
co_yield Instruction::Comment(" global read inc B loopL ");
co_yield GlobalReadInc(sgprSrdB, sgprShadowLimitB, sgprGlobalReadIncsB);
co_yield Instruction::Comment("****************************************\n"
                              " End setupNewTile, isPap=False             \n"
                              "****************************************\n");
co_yield Instruction::Label(label_0);
co_yield m_context->copier()->copy(sgprSrdD->subset({0}), D->subset({0}), " init SRD base address (lower)");
co_yield m_context->copier()->copy(sgprSrdD->subset({1}), D->subset({1}), " init SRD base address (upper) + other fields");
co_yield m_context->copier()->copy(sgprSrdD->subset({2}), Register::Value::Literal(2147483648), "0x80000000");
co_yield m_context->copier()->copy(sgprSrdD->subset({3}), Srd127_96, " Set bits 127_96 in post-loop SRD");
co_yield m_context->copier()->copy(sgprSrdC->subset({0}), C->subset({0}), " init SRD base address (lower)");
co_yield m_context->copier()->copy(sgprSrdC->subset({1}), C->subset({1}), " init SRD base address (upper) + other fields");
co_yield m_context->copier()->copy(sgprSrdC->subset({2}), Register::Value::Literal(2147483648), "0x80000000");
co_yield m_context->copier()->copy(sgprSrdC->subset({3}), Srd127_96, " Set bits 127_96 in post-loop SRD");
{
auto stempProd = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::UInt32, 1);
auto stemp = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Int64, 1);
co_yield generateOp<Expression::Multiply>(stempProd, macroTile1, sgprWorkGroup1); //" wg1*MT1"
co_yield generateOp<Expression::Multiply>(stemp, stempProd, sgprStrideC1J); //" CScale s62 by Stride"
co_yield generateOp<Expression::ShiftL>(stemp, stemp, Register::Value::Literal(2)); //" scale by bpe"
co_yield generateOp<Expression::Add>(sgprSrdC, C, stemp);
co_yield generateOp<Expression::Multiply>(stemp, stempProd, sgprStrideD1J); //" CScale s62 by Stride"
co_yield generateOp<Expression::ShiftL>(stemp, stemp, Register::Value::Literal(2)); //" scale by bpe"
co_yield generateOp<Expression::Add>(sgprSrdD, D, stemp);
co_yield generateOp<Expression::Multiply>(stemp, sgprWorkGroup2, sgprStrideCK); //"CScale s[sgprWorkGroup2] by Stride"
co_yield generateOp<Expression::ShiftL>(stemp, stemp, Register::Value::Literal(2)); //" scale by bpe"
co_yield generateOp<Expression::Add>(sgprSrdC->subset({0,1}), sgprSrdC->subset({0,1}), stemp);
co_yield generateOp<Expression::Multiply>(stemp, sgprWorkGroup2, sgprStrideDK); //"CScale s[sgprWorkGroup2] by Stride"
co_yield generateOp<Expression::ShiftL>(stemp, stemp, Register::Value::Literal(2)); //" scale by bpe"
co_yield generateOp<Expression::Add>(sgprSrdD->subset({0,1}), sgprSrdD->subset({0,1}), stemp);
}
co_yield Instruction::Comment(" initC: remove C-tile 0-0 from pool ");
co_yield Instruction::Comment(" initC: remove AB-tile 0-4 from pool ");
// co_yield fmm->zero(accDestination);
for(size_t i = 0; i < accDestination->valueCount(); ++i)
{
    co_yield_(Instruction("v_accvgpr_write",
                                  {accDestination->subset({i})},
                                  {Register::Value::Label("0x0")},
                                  {},
                                  " Set accumulator to 0"));
}
co_yield generateOp<Expression::Equal>(nullptr, sgprLoopCounterL, Register::Value::Literal(0));// " at last iteration?"
co_yield Instruction::Comment(" after InitC, skip to end of prefetch last iter if numIter==0 ");
co_yield m_context->brancher()->branchIfZero(label_1, m_context->getSCC(), " Only branch on scc1");
co_yield m_context->brancher()->branch(label_5, " branch to LoopEndL_2");
co_yield Instruction::Label(label_1);
co_yield Instruction::Comment(" local write a ");
co_yield m_context->mem()->store(MemoryInstructions::MemoryKind::Local, vgprLocalWriteAddrA, vgprG2LA, Register::Value::Literal(0), 4, " lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0");
co_yield Instruction::Comment(" local write b ");
co_yield m_context->mem()->store(MemoryInstructions::MemoryKind::Local, vgprLocalWriteAddrB, vgprG2LB, Register::Value::Literal(0), 4, " lwoB_0_0_0_0 = (0*LSCB) + (0*LSPB)(*MT1J+PAD) = 0");
co_yield Instruction::Comment(" local write swap a ");
co_yield generateOp<Expression::BitwiseXor>(vgprLocalWriteAddrA, Register::Value::Literal(2048), vgprLocalWriteAddrA); //" swap Red Blk"
co_yield Instruction::Comment(" local write swap b ");
co_yield generateOp<Expression::BitwiseXor>(vgprLocalWriteAddrB, Register::Value::Literal(2048), vgprLocalWriteAddrB); //" swap Red Blk"
co_yield Instruction::Comment("****************************************\n"
                              " Unrolled Loop(s) - Begin               \n"
                              "****************************************\n");
co_yield Instruction::Label(label_2);
co_yield generateOp<Expression::LessThanEqual>(nullptr, sgprLoopCounterL, Register::Value::Literal(0));// " LoopCounterL < EndCounter"
co_yield m_context->brancher()->branchIfNonZero(label_5, m_context->getSCC(), " do not enter LoopL");
co_yield Instruction::Label(label_3);
co_yield Instruction::Comment("****************************************\n"
                              " Unrolled Loop 1/1 - Begin              \n"
                              "****************************************\n");
co_yield Instruction::Label(label_4);
co_yield Instruction::Comment(" Begin Each Unroll: Check VGPR.checkin for INT8 LW ");
co_yield generateOp<Expression::Equal>(nullptr, sgprLoopCounterL, Register::Value::Literal(1));// " is this the last iteration"
co_yield m_context->copier()->conditionalCopy(sgprSrdA->subset({2}), Register::Value::Literal(0), " Set limit to 0 for last iteration");
co_yield m_context->copier()->conditionalCopy(sgprSrdB->subset({2}), Register::Value::Literal(0), " Set limit to 0 for last iteration");
co_yield Instruction::Comment(" iter 0 ");
co_yield_(Instruction("buffer_load_dword", {vgprG2LA}, {vgprGlobalReadOffsetA, sgprSrdA, Register::Value::Literal(0)}, {"offen", "offset:0"}, " G -> Reg 0_0_0_0"));
co_yield_(Instruction("buffer_load_dword", {vgprG2LB}, {vgprGlobalReadOffsetB, sgprSrdB, Register::Value::Literal(0)}, {"offen", "offset:0"}, " G -> Reg 0_0_0_0"));
co_yield Instruction::Comment(" global read inc A loopL ");
co_yield GlobalReadInc(sgprSrdA, sgprShadowLimitA, sgprGlobalReadIncsA);
co_yield Instruction::Comment(" global read inc B loopL ");
co_yield GlobalReadInc(sgprSrdB, sgprShadowLimitB, sgprGlobalReadIncsB);
{
auto vgprValuA_X0_I0 = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
auto vgprValuA_X1_I0 = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
auto vgprValuB_X0_I0 = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
auto vgprValuB_X1_I0 = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Float, 1);
co_yield m_context->mem()->barrier(); //"4sync for global read"
co_yield Instruction::Comment(" local read a ");
co_yield m_context->mem()->load(MemoryInstructions::MemoryKind::Local, vgprValuA_X0_I0, vgprLocalReadAddrA, Register::Value::Literal(0), 4, "L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0");
co_yield Instruction::Comment(" local read b ");
co_yield m_context->mem()->load(MemoryInstructions::MemoryKind::Local, vgprValuB_X0_I0, vgprLocalReadAddrB, Register::Value::Literal(0), 4, "L -> Reg lro=0 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=0 iui=0");
co_yield Instruction::Comment(" local read increment a \n"
                              " N/A, lro->128 \n"
                              " self.localReadDoCntA 1 self.localReadDoCntB 1 \n"
                              " local read increment b \n"
                              " N/A, lro->128 \n"
                              " self.localReadDoCntA 1 self.localReadDoCntB 1 \n"
                              " sched write - iter 0 writesPerItem=1 \n");
co_yield m_context->mem()->store(MemoryInstructions::MemoryKind::Local, vgprLocalWriteAddrA, vgprG2LA, Register::Value::Literal(0), 4, " lwoA_0_0_0_0 = (0*LSCA) + (0*LSPA)(*MT0I+PAD) = 0");
co_yield fmm->mul(accDestination, vgprValuA_X0_I0, vgprValuB_X0_I0, accDestination, 32, 32, 2, 1);
co_yield Instruction::Comment(" numPrefetchIter=0 \n"
                              " dataAtIterA=0 numReadsIterA=1 skipReadsIterA=0 readsPerIterA=1 \n"
                              " dataAtIterB=0 numReadsIterB=1 skipReadsIterB=0 readsPerIterB=1 \n"
                              " iter 1 (reset local read pointers iteration)  (swap and reset local write pointers iteration)  (swap local read pointers iteration)  \n");
co_yield Instruction::Comment(" local read a ");
co_yield m_context->mem()->load(MemoryInstructions::MemoryKind::Local, vgprValuA_X1_I0, vgprLocalReadAddrA, Register::Value::Literal(512), 4, "L -> Reg lro=128 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0");
co_yield Instruction::Comment(" local read b ");
co_yield m_context->mem()->load(MemoryInstructions::MemoryKind::Local, vgprValuB_X1_I0, vgprLocalReadAddrB, Register::Value::Literal(512), 4, "L -> Reg lro=128 swapByteOffset=0 ti=64 vIdx=0 rIdx=0 oIdx=0 buffer=1 iui=0");
co_yield Instruction::Comment(" sched write - iter 1 writesPerItem=1 ");
co_yield m_context->mem()->store(MemoryInstructions::MemoryKind::Local, vgprLocalWriteAddrB, vgprG2LB, Register::Value::Literal(0), 4, " lwoB_0_0_0_0 = (0*LSCB) + (0*LSPB)(*MT1J+PAD) = 0");
co_yield Instruction::Comment(" local write swap offsets a ");
co_yield generateOp<Expression::BitwiseXor>(vgprLocalWriteAddrA, Register::Value::Literal(2048), vgprLocalWriteAddrA); //" swap Red Blk"
co_yield Instruction::Comment(" local write swap offsets b ");
co_yield generateOp<Expression::BitwiseXor>(vgprLocalWriteAddrB, Register::Value::Literal(2048), vgprLocalWriteAddrB); //" swap Red Blk"
co_yield Instruction::Comment(" local read swap offsets a ");
co_yield generateOp<Expression::BitwiseXor>(vgprLocalReadAddrA, Register::Value::Literal(2048), vgprLocalReadAddrA); //" swap Red Blk"
co_yield Instruction::Comment(" local read swap offsets b ");
co_yield generateOp<Expression::BitwiseXor>(vgprLocalReadAddrB, Register::Value::Literal(2048), vgprLocalReadAddrB); //" swap Red Blk"
co_yield Instruction::Comment(" local read init pointers a ");
co_yield Instruction::Comment(" localReadInitPointers ");
co_yield Instruction::Comment(" local read init pointers b ");
co_yield Instruction::Comment(" localReadInitPointers ");
co_yield fmm->mul(accDestination, vgprValuA_X1_I0, vgprValuB_X1_I0, accDestination, 32, 32, 2, 1);
co_yield Instruction::Comment(" numPrefetchIter=0 ");
co_yield Instruction::Comment(" dataAtIterA=1 numReadsIterA=2 skipReadsIterA=0 readsPerIterA=1 ");
co_yield Instruction::Comment(" dataAtIterB=1 numReadsIterB=2 skipReadsIterB=0 readsPerIterB=1 ");
}
co_yield Instruction::Comment("****************************************\n"
                              " Unrolled Loop - End                    \n"
                              "****************************************\n"
                              " closeLoop loopL finalLoop=1 tailLoop=0 \n");
co_yield generateOp<Expression::Subtract>(sgprLoopCounterL, sgprLoopCounterL, Register::Value::Literal(1)); //" dec counterL"
co_yield generateOp<Expression::Equal>(nullptr, sgprLoopCounterL, Register::Value::Literal(0)); //" counterL==0"
co_yield m_context->brancher()->branchIfZero(label_3, m_context->getSCC(), " restart LoopL");
co_yield Instruction::Label(label_5);
co_yield Instruction::Comment(" Before NLL: Check VGPR.checkin for INT8 LW ");
co_yield Instruction::Label(label_6);
co_yield Instruction::Comment(" endSummation: add vgpr [0...11) to pool ");
co_yield Instruction::Comment(" Mapping of Acc register -> C Vgpr register ");
co_yield Instruction::Comment(" not-LocalSplitU: global write indices ");
{
auto stemp = std::make_shared<Register::Value>(m_context, Register::Type::Scalar, DataType::Int32, 1);
auto vtemp1 = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
auto vtemp2 = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
auto vtemp3 = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
auto vtemp4 = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
auto vtemp5 = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
auto vtemp6 = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 1);
auto vDestination = std::make_shared<Register::Value>(m_context, Register::Type::Vector, DataType::Int32, 16, Register::AllocationOptions::FullyContiguous());
co_yield Instruction::Comment(" computeStoreVgprs ");
co_yield generateOp<Expression::LogicalShiftR>(vtemp1, vgprSerial, Register::Value::Literal(6)); //" vtemp1 = v[vgprSerial] / 64"
co_yield generateOp<Expression::LogicalShiftR>(vtemp6, vtemp1, Register::Value::Literal(1)); //" vtemp6 = vtemp1 / 2"
co_yield generateOp<Expression::Multiply>(vtemp6, Register::Value::Literal(32), vtemp6); //" wave coordination offset 1"
co_yield generateOp<Expression::BitwiseAnd>(vtemp2, Register::Value::Literal(31), vgprSerial); //" vtemp2 = v[vgprSerial] % 32"
co_yield generateOp<Expression::Add>(vtemp6, vtemp2, vtemp6); //" coordination 1 = wave_id1 + tid1"
//TODO: MISTAKE? co_yield Instruction("v_mul_lo_u32", {v2}, {vtemp6, sgprStrideC1J}, {}, "  offset 1");
co_yield generateOp<Expression::Multiply>(vtemp5, vtemp6, sgprStrideD1J); //"  offset 1"
co_yield generateOp<Expression::BitwiseAnd>(vtemp2, Register::Value::Literal(1), vtemp1); //" vtemp2 = vtemp1 % 2"
co_yield generateOp<Expression::Multiply>(vtemp2, Register::Value::Literal(32), vtemp2); //" wave coordination offset 0"
co_yield generateOp<Expression::BitwiseAnd>(vtemp4, Register::Value::Literal(63), vgprSerial); //" vtemp4 = v[vgprSerial] % 64"
co_yield generateOp<Expression::LogicalShiftR>(vtemp4, vtemp4, Register::Value::Literal(5)); //" vtemp4 = vtemp4 / 32"
//" thread0 * continuous_output"
//" coordination 0 = wave_id0 + tid0"
co_yield generateOp<Expression::ShiftLAdd>(vtemp4, vtemp4, Register::Value::Literal(2), vtemp2);
co_yield generateOp<Expression::Multiply>(stemp, macroTile0, sgprWorkGroup0); //" wgp0 * MT0"
co_yield generateOp<Expression::Add>(vtemp4, stemp, vtemp4); //" coord 0 = (tid0/MI_m)*4 + waveG0*MIB_m + MT0*SG0"
co_yield generateOp<Expression::Multiply>(stemp, macroTile1, sgprWorkGroup1); //" wgp1 * MT1"
co_yield generateOp<Expression::Add>(vtemp6, stemp, vtemp6); //" coord 1 = (tid0%MI_m) + waveG1*MIB_n + MT1*SG1"
co_yield Instruction::Comment(" not-LocalSplitU: global write ");
co_yield Instruction::Label(label_7);
co_yield Instruction::Comment(" edge=0, allocate 2 sgpr. perBatchTmpS=2 perBatchMaskS=0 perElementMaskS=0 elementsPerBatch=4 \n"
                              " optSingleColVgpr=1 optSharedColVgpr=0 optSGPRUsage=BufferLoad_Mask optSrdIncForRow=1 \n"
                              "****************************************\n"
                              " Global Write Batch #0 (d1,d0,vc1,vc0) = \n"
                              "    (0,0,0,0:vw4); (0,1,0,0:vw4); (0,2,0,0:vw4); (0,3,0,0:vw4) \n"
                              "****************************************\n"
                              " calc coords, apply mask, and issue loads (if necessary) \n"
                              " (d1,vc1,d0,vc0)=(0,0,0,0) \n"
                              " (d1,vc1,d0,vc0)=(0,0,1,0) \n"
                              " (d1,vc1,d0,vc0)=(0,0,2,0) \n"
                              " (d1,vc1,d0,vc0)=(0,0,3,0) \n");
co_yield_(Instruction("v_add_lshl_u32", {vtemp3}, {vtemp5, vtemp4, Register::Value::Literal(2)}, {}, " optSingleColVgpr scaleToBpe: sharedAddrVgpr <- cinRowPtr + coord0, scaled by BPE. BSHERE:coord0=0, coord0Vgpr=0"));
co_yield m_context->copier()->copy(vDestination, accDestination, "copy acc to vreg");
co_yield Instruction::Nop(1); //" 2 wait states required before reading vgpr"
co_yield Instruction::Comment(" rC *= alpha batchElements=[(0, 0, 0, 0), (0, 1, 0, 0), (0, 2, 0, 0), (0, 3, 0, 0)] ");
co_yield Instruction::Comment(" apply mask, calc new C and issue writes ");
for(int i = 0; i < 16; i+=4)
{
    co_yield_(Instruction("buffer_store_dwordx4", {vDestination->subset({i, i+1, i+2, i+3})}, {vtemp3, sgprSrdD, Register::Value::Literal(0)}, {"offen", concatenate("offset:", i * 8)}, " store D"));
}
}
    // clang-format on
}
