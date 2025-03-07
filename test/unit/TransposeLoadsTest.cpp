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

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/DataTypes/DataTypes_Utils.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Utilities/Generator.hpp>

#include "GPUContextFixture.hpp"
#include "Utilities.hpp"

using namespace rocRoller;

namespace TransposeLoadsTest
{
    template <typename ElementType, typename PackType, bool unalignedVGPRs>
    void generateTransposeLoad(rocRoller::ContextPtr m_context, const int MN, const int K)
    {
        constexpr uint elementBits = TypeInfo<ElementType>::ElementBits;
        static_assert(((unalignedVGPRs && elementBits == 6) || !unalignedVGPRs)
                      && "Unaligned VGPRs are only allowed with B6 tranpose loads!");
        // DS_READ_B96_TR_B6 requires 128b-aligned addresses
        constexpr uint extraLdsBytes        = (elementBits == 6) ? (128 - 96) / 8 : 0;
        const DataType elementDataType      = TypeInfo<ElementType>::Var.dataType;
        const DataType packDataType         = TypeInfo<PackType>::Var.dataType;
        const uint     workitemCountX       = 64u;
        const uint     numWorkitemsPerWave  = workitemCountX;
        const auto     packDataTypeInfo     = DataTypeInfo::Get(packDataType);
        const uint     packBytes            = packDataTypeInfo.elementBits / 8;
        const uint     bytesPerTrLoad       = bitsPerTransposeLoad(elementBits) / 8;
        const uint     bytesPerWorkitem     = bytesPerTrLoad * /*numberOfLDSTrLoads*/ 2;
        const uint     bytesPerWord         = 4;
        const uint     registerCountPerLoad = bytesPerTrLoad / packBytes;
        const uint     threadTrLoadOffset   = MN * (bytesPerTrLoad + extraLdsBytes);

        auto k = m_context->kernel();

        auto one                = Expression::literal(1);
        auto workitemCountXExpr = Expression::literal(workitemCountX);

        k->setKernelName(toString(elementDataType) + "TransposeLoad");
        k->setKernelDimensions(1);
        k->setWorkgroupSize({workitemCountX, 1, 1});
        k->setWorkitemCount({workitemCountXExpr, one, one});

        auto command = std::make_shared<Command>();

        auto resultTag  = command->allocateTag();
        auto resultExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::UInt32, PointerType::PointerGlobal}, resultTag, ArgumentType::Value));

        auto aTag  = command->allocateTag();
        auto aExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::UInt32, PointerType::PointerGlobal}, aTag, ArgumentType::Value));

        auto trLoadIdxTag  = command->allocateTag();
        auto trLoadIdxExpr = std::make_shared<Expression::Expression>(command->allocateArgument(
            {DataType::UInt32, PointerType::PointerGlobal}, trLoadIdxTag, ArgumentType::Value));

        k->addArgument({"result",
                        {DataType::UInt32, PointerType::PointerGlobal},
                        DataDirection::WriteOnly,
                        resultExpr});

        k->addArgument(
            {"a", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::ReadOnly, aExpr});

        k->addArgument({"trLoadIdx",
                        {DataType::UInt32, PointerType::PointerGlobal},
                        DataDirection::ReadOnly,
                        trLoadIdxExpr});

        m_context->schedule(k->preamble());
        m_context->schedule(k->prolog());

        auto kb = [&]() -> Generator<Instruction> {
            Register::ValuePtr sResult, sA, sTrLoadIdx;
            co_yield m_context->argLoader()->getValue("result", sResult);
            co_yield m_context->argLoader()->getValue("a", sA);
            co_yield m_context->argLoader()->getValue("trLoadIdx", sTrLoadIdx);

            auto vWorkitemX = m_context->kernel()->workitemIndex()[0];

            auto vLinearWorkitemOffset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);
            auto vLinearWordOffset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);
            auto vTransposeOffset = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);

            auto vBytesPerWord = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);
            auto vBytesPerWorkitem = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);
            auto vBytesPerTrLoad = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);

            auto vTransposeWorkitemIdx = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);

            auto                        FC{Register::AllocationOptions::FullyContiguous()};
            Register::AllocationOptions vA0TRegAllocOptions;
            if constexpr(elementBits == 6 && unalignedVGPRs)
                vA0TRegAllocOptions = {.alignment = 3};
            else
                vA0TRegAllocOptions = FC;

            auto vA0 = Register::Value::Placeholder(
                m_context, Register::Type::Vector, packDataType, registerCountPerLoad, FC);

            auto vA1 = Register::Value::Placeholder(
                m_context, Register::Type::Vector, packDataType, registerCountPerLoad, FC);

            auto vA0T = Register::Value::Placeholder(m_context,
                                                     Register::Type::Vector,
                                                     packDataType,
                                                     registerCountPerLoad,
                                                     vA0TRegAllocOptions);

            auto vA1T = Register::Value::Placeholder(
                m_context, Register::Type::Vector, packDataType, registerCountPerLoad, FC);

            auto vAPtr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::UInt32, PointerType::PointerGlobal},
                                               1,
                                               FC);

            auto vResultPtr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::UInt32, PointerType::PointerGlobal},
                                               1,
                                               FC);
            auto vTrLoadIdxAddr
                = Register::Value::Placeholder(m_context,
                                               Register::Type::Vector,
                                               {DataType::UInt32, PointerType::PointerGlobal},
                                               1,
                                               FC);

            Register::ValuePtr lds;
            if constexpr(elementBits == 6)
            {
                // pad each row so each workitem points to 128b instead of 96b.
                lds = Register::Value::AllocateLDS(m_context,
                                                   DataType::FP8,
                                                   (elementBits * MN * K) / 8
                                                       + MN * (K / 16) * extraLdsBytes);
            }
            else
            {
                lds = Register::Value::AllocateLDS(m_context, elementDataType, MN * K);
            }
            auto vLDSBasePtr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);
            auto vLDSPtr = Register::Value::Placeholder(
                m_context, Register::Type::Vector, DataType::UInt32, 1);
            co_yield m_context->copier()->copy(
                vLDSBasePtr, Register::Value::Literal(lds->getLDSAllocation()->offset()));

            co_yield vA0->allocate();
            co_yield vA1->allocate();
            co_yield vA0T->allocate();
            co_yield vA1T->allocate();

            co_yield m_context->copier()->copy(vAPtr, sA);
            co_yield m_context->copier()->copy(vResultPtr, sResult);
            co_yield m_context->copier()->copy(vTrLoadIdxAddr, sTrLoadIdx);

            co_yield m_context->copier()->fill(vBytesPerWord,
                                               Register::Value::Literal(bytesPerWord));

            co_yield m_context->copier()->fill(vBytesPerWorkitem,
                                               Register::Value::Literal(bytesPerWorkitem));

            co_yield m_context->copier()->fill(vBytesPerTrLoad,
                                               Register::Value::Literal(bytesPerTrLoad));

            co_yield generateOp<Expression::Multiply>(
                vLinearWorkitemOffset, vWorkitemX, vBytesPerWorkitem);

            co_yield generateOp<Expression::Add>(vAPtr, vAPtr, vLinearWorkitemOffset);
            if constexpr(elementBits == 6)
            {

                auto v256Bits32Bytes = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);
                auto v256Bits32BytesOffset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);
                co_yield m_context->copier()->fill(
                    v256Bits32Bytes,
                    Register::Value::Literal((bytesPerTrLoad + extraLdsBytes) * 2));
                co_yield generateOp<Expression::Multiply>(
                    v256Bits32BytesOffset, vWorkitemX, v256Bits32Bytes);
                co_yield generateOp<Expression::Add>(vLDSPtr, vLDSBasePtr, v256Bits32BytesOffset);
            }
            else
            {
                co_yield generateOp<Expression::Add>(vLDSPtr, vLDSBasePtr, vLinearWorkitemOffset);
            }

            co_yield m_context->mem()->loadGlobal(vA0, vAPtr, /*offset*/ 0, bytesPerTrLoad);
            co_yield m_context->mem()->loadGlobal(vA1, vAPtr, bytesPerTrLoad, bytesPerTrLoad);
            co_yield m_context->mem()->storeLocal(vLDSPtr, vA0, /*offset*/ 0, bytesPerTrLoad);
            co_yield m_context->mem()->storeLocal(
                vLDSPtr, vA1, bytesPerTrLoad + extraLdsBytes, bytesPerTrLoad);
            co_yield m_context->mem()->barrier();

            co_yield generateOp<Expression::Multiply>(vLinearWordOffset, vWorkitemX, vBytesPerWord);
            co_yield generateOp<Expression::Add>(vTrLoadIdxAddr, vTrLoadIdxAddr, vLinearWordOffset);
            co_yield m_context->mem()->loadGlobal(
                vTransposeWorkitemIdx, vTrLoadIdxAddr, /*offset*/ 0, bytesPerWord);
            if constexpr(elementBits == 6)
            {
                auto v128Bits16Bytes = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);
                co_yield m_context->copier()->fill(
                    v128Bits16Bytes, Register::Value::Literal(bytesPerTrLoad + extraLdsBytes));
                co_yield generateOp<Expression::Multiply>(
                    vTransposeOffset, vTransposeWorkitemIdx, v128Bits16Bytes);
            }
            else
            {
                co_yield generateOp<Expression::Multiply>(
                    vTransposeOffset, vTransposeWorkitemIdx, vBytesPerTrLoad);
            }

            co_yield generateOp<Expression::Add>(vLDSPtr, vLDSBasePtr, vTransposeOffset);

            co_yield m_context->mem()->transposeLoadLocal(
                vA0T, vLDSPtr, /*offset*/ 0, bytesPerTrLoad + extraLdsBytes, elementBits);
            co_yield m_context->mem()->transposeLoadLocal(vA1T,
                                                          vLDSPtr,
                                                          /*offset*/ threadTrLoadOffset,
                                                          bytesPerTrLoad + extraLdsBytes,
                                                          elementBits);

            co_yield m_context->mem()->barrier();
            if constexpr(elementBits == 6)
            {
                auto vLinearTRLoadBytesOffset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);
                co_yield generateOp<Expression::Multiply>(
                    vLinearTRLoadBytesOffset, vWorkitemX, vBytesPerTrLoad);

                co_yield generateOp<Expression::Add>(
                    vResultPtr, vResultPtr, vLinearTRLoadBytesOffset);

                // copy unaligned VGPRs to aligned ones
                co_yield m_context->copier()->copy(vA0, vA0T);
                co_yield m_context->copier()->copy(vA1, vA1T);

                co_yield m_context->mem()->storeGlobal(vResultPtr,
                                                       vA0,
                                                       /*offset*/ 0,
                                                       bytesPerTrLoad);
                co_yield m_context->mem()->storeGlobal(vResultPtr,
                                                       vA1,
                                                       /*offset*/ numWorkitemsPerWave
                                                           * bytesPerTrLoad,
                                                       bytesPerTrLoad);
            }
            else
            {
                co_yield generateOp<Expression::Add>(vResultPtr, vResultPtr, vLinearWordOffset);

                const uint regCount = bitsPerTransposeLoad(elementBits) / 32;
                for(uint regIdx = 0; regIdx < regCount; regIdx++)
                {
                    co_yield m_context->mem()->storeGlobal(vResultPtr,
                                                           vA0T->subset({regIdx}),
                                                           /*offset*/ regIdx * numWorkitemsPerWave
                                                               * bytesPerWord,
                                                           bytesPerWord);
                    co_yield m_context->mem()->storeGlobal(vResultPtr,
                                                           vA1T->subset({regIdx}),
                                                           /*offset*/ (regCount + regIdx)
                                                               * numWorkitemsPerWave * bytesPerWord,
                                                           bytesPerWord);
                }
            }
        };

        m_context->schedule(kb());
        m_context->schedule(k->postamble());
        m_context->schedule(k->amdgpu_metadata());
    }

    template <typename StoredAsType, uint elementBits>
    std::vector<uint32_t> pack(const std::vector<StoredAsType>& unpacked)
    {
        static_assert(
            (elementBits == 4 || elementBits == 6 || elementBits == 8 || elementBits == 16)
            && "Only 4, 6, 8, and 16 bits are supported!");

        if constexpr(elementBits == 4)
        {
            return packFP4x8(unpacked);
        }

        if constexpr(elementBits == 6)
        {
            return packF6x16(unpacked);
        }

        if constexpr(elementBits == 8 || elementBits == 16)
        {
            std::vector<uint32_t> packed(unpacked.size() * sizeof(StoredAsType) / sizeof(uint32_t));
            std::memcpy(packed.data(), unpacked.data(), sizeof(uint32_t) * packed.size());
            return packed;
        }
    }

    template <typename StoredAsType, uint elementBits>
    std::vector<StoredAsType> unpack(const std::vector<uint32_t>& packed)
    {
        static_assert(
            (elementBits == 4 || elementBits == 6 || elementBits == 8 || elementBits == 16)
            && "Only 4, 6, 8, and 16 bits are supported!");

        if constexpr(elementBits == 4)
        {
            return unpackFP4x8(packed);
        }

        if constexpr(elementBits == 6)
        {
            return unpackF6x16(packed);
        }

        if constexpr(elementBits == 8 || elementBits == 16)
        {
            std::vector<StoredAsType> unpacked(packed.size() * sizeof(uint32_t)
                                               / sizeof(StoredAsType));
            std::memcpy(unpacked.data(), packed.data(), sizeof(StoredAsType) * unpacked.size());
            return unpacked;
        }
    }

    template <typename ElementType, typename PackType, typename StoredAsType, bool unalignedVGPRs>
    void executeTransposeLoad(rocRoller::ContextPtr m_context, const int MN, const int K)
    {
        constexpr uint elementBits = TypeInfo<ElementType>::ElementBits;

        std::vector<StoredAsType> data(MN * K);
        for(int i = 0; i < MN; i++)
            for(int j = 0; j < K; j++)
                data[i * K + j] = rand() % 10;

        auto packedData = pack<StoredAsType, elementBits>(data);
        ASSERT_TRUE(packedData.size() > 0);

        std::vector<uint32_t> result(packedData.size());

        std::vector<uint32_t> trLoadIdx(64);
        {
            const int factor = 2;
            const int NX1    = MN / 16;
            const int NX0    = 4 / NX1;
            // each thread points to 64b or 96b
            const int NY0 = bitsPerTransposeLoad(elementBits) / elementBits;
            const int NY1 = 16 / NY0;
            for(int x0 = 0; x0 < NX0; x0++)
                for(int x1 = 0; x1 < NX1; x1++)
                    for(int y0 = 0; y0 < NY0; y0++)
                        for(int y1 = 0; y1 < NY1; y1++)
                        {
                            trLoadIdx[NX1 * NY0 * NY1 * x0 + NY0 * NY1 * x1 + NY1 * y0 + y1]
                                = factor * NX1 * NY0 * NY1 * x0 + NX1 * NY1 * y0 + NY1 * x1 + y1;
                        }
        }

        generateTransposeLoad<ElementType, PackType, unalignedVGPRs>(m_context, MN, K);
        CommandKernel commandKernel;
        commandKernel.setContext(m_context);

        auto d_a         = make_shared_device(packedData);
        auto d_result    = make_shared_device<uint32_t>(result.size());
        auto d_trLoadIdx = make_shared_device(trLoadIdx);

        KernelArguments runtimeArgs;
        runtimeArgs.append("result", d_result.get());
        runtimeArgs.append("a", d_a.get());
        runtimeArgs.append("trLoadIdx", d_trLoadIdx.get());

        commandKernel.launchKernel(runtimeArgs.runtimeArguments());

        ASSERT_THAT(
            hipMemcpy(
                result.data(), d_result.get(), sizeof(uint32_t) * result.size(), hipMemcpyDefault),
            HasHipSuccess(0));

        auto result_unpacked = unpack<StoredAsType, elementBits>(result);
        ASSERT_TRUE(packedData.size() > 0);

        {
            int NY0 = 4;
            int NY2 = 32 / elementBits;
            if constexpr(elementBits == 6)
            {
                NY0 = 2;
                NY2 = 96 / elementBits;
            }

            const int NY1 = K / NY2 / NY0;
            const int NX0 = MN / 16;

            for(int y0 = 0; y0 < NY0; y0++)
                for(int y1 = 0; y1 < NY1; y1++)
                    for(int x0 = 0; x0 < NX0; x0++)
                        for(int x1 = 0; x1 < 4; x1++)
                            for(int x2 = 0; x2 < 4; x2++)
                                for(int y2 = 0; y2 < NY2; y2++)
                                {
                                    ASSERT_EQ(data[y1 * NY0 * NY2 * NX0 * 16 + y0 * NY2 * NX0 * 16
                                                   + y2 * NX0 * 16 + x0 * 16 + x1 * 4 + x2],
                                              result_unpacked[y0 * NY1 * NX0 * NY2 * 16
                                                              + y1 * NX0 * NY2 * 16 + x0 * 16 * NY2
                                                              + x1 * 4 * NY2 + x2 * NY2 + y2]);
                                }
        }
    }

    template <typename ElementType, uint dwordX>
    void checkGeneratedCode(rocRoller::ContextPtr m_context)
    {
        constexpr uint elementBits     = TypeInfo<ElementType>::ElementBits;
        const uint     dsReadWriteBits = bitsPerTransposeLoad(elementBits);
        std::string    mnemonic        = transposeLoadMnemonic(elementBits);

        // FIXME: waiting for std::format :(
        std::string globalLoadDWordX{"global_load_dwordx" + std::to_string(dwordX) + " "};
        std::string dsWriteBX{"ds_write_b" + std::to_string(dsReadWriteBits) + " "};

        std::string code = m_context->instructions()->toString();

        EXPECT_EQ(countSubstring(code, mnemonic), 2);
        EXPECT_EQ(countSubstring(code, "global_load_dword"), 3);
        EXPECT_EQ(countSubstring(code, globalLoadDWordX), 2);

        EXPECT_EQ(countSubstring(code, "ds_write_b"), 2);
        EXPECT_EQ(countSubstring(code, dsWriteBX), 2);
    }

    struct TransposeInstructionsTest : public CurrentGPUContextFixture
    {
    };

    TEST_F(TransposeInstructionsTest, B16Transpose16x32GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDSReadTransposeB16);
        executeTransposeLoad<Half, Halfx2, uint16_t, /*unalignedVGPRs*/ false>(m_context, 16, 32);
    }

    TEST_F(TransposeInstructionsTest, B16Transpose16x32GenTest)
    {
        generateTransposeLoad<Half, Halfx2, /*unalignedVGPRs*/ false>(m_context, 16, 32);
        checkGeneratedCode<Half, /*dwordx*/ 2>(m_context);
    }

    TEST_F(TransposeInstructionsTest, B16Transpose32x16GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDSReadTransposeB16);
        executeTransposeLoad<Half, Halfx2, uint16_t, /*unalignedVGPRs*/ false>(m_context, 32, 16);
    }

    TEST_F(TransposeInstructionsTest, B16Transpose32x16GenTest)
    {
        generateTransposeLoad<Half, Halfx2, /*unalignedVGPRs*/ false>(m_context, 32, 16);
        checkGeneratedCode<Half, /*dwordx*/ 2>(m_context);
    }

    TEST_F(TransposeInstructionsTest, B8Transpose16x64GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDSReadTransposeB8);
        executeTransposeLoad<FP8, FP8x4, uint8_t, /*unalignedVGPRs*/ false>(m_context, 16, 64);
    }

    TEST_F(TransposeInstructionsTest, B8Transpose16x64GenTest)
    {
        generateTransposeLoad<FP8, FP8x4, /*unalignedVGPRs*/ false>(m_context, 16, 64);
        checkGeneratedCode<FP8, /*dwordx*/ 2>(m_context);
    }

    TEST_F(TransposeInstructionsTest, B8Transpose32x32GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDSReadTransposeB8);
        executeTransposeLoad<FP8, FP8x4, uint8_t, /*unalignedVGPRs*/ false>(m_context, 32, 32);
    }

    TEST_F(TransposeInstructionsTest, B8Transpose32x32GenTest)
    {
        generateTransposeLoad<FP8, FP8x4, /*unalignedVGPRs*/ false>(m_context, 32, 32);
        checkGeneratedCode<FP8, /*dwordx*/ 2>(m_context);
    }

    TEST_F(TransposeInstructionsTest, B6AlignedVGPRsTranspose16x128GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDSReadTransposeB6);
        executeTransposeLoad<FP6, FP6x16, uint8_t, /*unalignedVGPRs*/ false>(m_context, 16, 128);
    }

    TEST_F(TransposeInstructionsTest, B6AlignedVGPRsTranspose16x128GenTest)
    {
        generateTransposeLoad<FP6, FP6x16, /*unalignedVGPRs*/ false>(m_context, 16, 128);
        checkGeneratedCode<FP6, /*dwordx*/ 3>(m_context);
    }

    TEST_F(TransposeInstructionsTest, B6AlignedVGPRsTranspose32x64GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDSReadTransposeB6);
        executeTransposeLoad<FP6, FP6x16, uint8_t, /*unalignedVGPRs*/ false>(m_context, 32, 64);
    }

    TEST_F(TransposeInstructionsTest, B6AlignedVGPRsTranspose32x64GenTest)
    {
        generateTransposeLoad<FP6, FP6x16, /*unalignedVGPRs*/ false>(m_context, 32, 64);
        checkGeneratedCode<FP6, /*dwordx*/ 3>(m_context);
    }

    TEST_F(TransposeInstructionsTest, B6UnalignedVGPRsTranspose16x128GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDSReadTransposeB6);
        executeTransposeLoad<FP6, FP6x16, uint8_t, /*unalignedVGPRs*/ true>(m_context, 16, 128);
    }

    TEST_F(TransposeInstructionsTest, B6UnalignedVGPRsTranspose16x128GenTest)
    {
        generateTransposeLoad<FP6, FP6x16, /*unalignedVGPRs*/ true>(m_context, 16, 128);
        checkGeneratedCode<FP6, /*dwordx*/ 3>(m_context);
    }

    TEST_F(TransposeInstructionsTest, B6UnalignedVGPRsTranspose32x64GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDSReadTransposeB6);
        executeTransposeLoad<FP6, FP6x16, uint8_t, /*unalignedVGPRs*/ true>(m_context, 32, 64);
    }

    TEST_F(TransposeInstructionsTest, B6UnalignedVGPRsTranspose32x64GenTest)
    {
        generateTransposeLoad<FP6, FP6x16, /*unalignedVGPRs*/ true>(m_context, 32, 64);
        checkGeneratedCode<FP6, /*dwordx*/ 3>(m_context);
    }

    TEST_F(TransposeInstructionsTest, B4Transpose16x128GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDSReadTransposeB4);
        executeTransposeLoad<FP4, FP4x8, uint8_t, /*unalignedVGPRs*/ false>(m_context, 16, 128);
    }

    TEST_F(TransposeInstructionsTest, B4Transpose16x128GenTest)
    {
        generateTransposeLoad<FP4, FP4x8, /*unalignedVGPRs*/ false>(m_context, 16, 128);
        checkGeneratedCode<FP4, /*dwordx*/ 2>(m_context);
    }

    TEST_F(TransposeInstructionsTest, B4Transpose32x64GPUTest)
    {
        REQUIRE_ARCH_CAP(GPUCapability::HasDSReadTransposeB4);
        executeTransposeLoad<FP4, FP4x8, uint8_t, /*unalignedVGPRs*/ false>(m_context, 32, 64);
    }

    TEST_F(TransposeInstructionsTest, B4Transpose32x64GenTest)
    {
        generateTransposeLoad<FP4, FP4x8, /*unalignedVGPRs*/ false>(m_context, 32, 64);
        checkGeneratedCode<FP4, /*dwordx*/ 2>(m_context);
    }
}
