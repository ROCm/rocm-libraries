/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025 AMD ROCm(TM) Software
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

#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CodeGen/Utils.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Operations/Command.hpp>

#include "CustomMatchers.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"
#include "common/Utilities.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers.hpp>

using namespace rocRoller;

namespace TransposeLoadsTest
{
    struct TransposeLoadKernel : public AssemblyTestKernel
    {
        using ExpressionFunc = std::function<Expression::ExpressionPtr(Expression::ExpressionPtr,
                                                                       Expression::ExpressionPtr)>;
        TransposeLoadKernel(ContextPtr context,
                            DataType   elementType,
                            bool       unalignedVGPRs,
                            uint       dwordX,
                            uint       MN,
                            uint       K)
            : AssemblyTestKernel(context)
            , m_elementType(elementType)
            , m_dwordX(dwordX)
            , m_unalignedVGPRs(unalignedVGPRs)
            , m_MN(MN)
            , m_K(K)
        {
            m_elementBits = DataTypeInfo::Get(m_elementType).elementBits;
        }

        void generate() override
        {
            AssertFatal(((m_unalignedVGPRs && m_elementBits == 6) || !m_unalignedVGPRs),
                        "Unaligned VGPRs are only allowed with B6 transpose loads!");

            auto const& arch = m_context->targetArchitecture();

            // DS_READ_B96_TR_B6 requires 128b-aligned addresses
            const uint     extraLdsBytes   = (m_elementBits == 6) ? (128 - 96) / 8 : 0;
            const DataType elementDataType = m_elementType;
            const DataType packDataType
                = DataTypeInfo::Get(m_elementType).packedVariableType()->dataType;
            const uint workitemCountX       = 64u;
            const uint numWorkitemsPerWave  = workitemCountX;
            const uint packBytes            = DataTypeInfo::Get(packDataType).elementBits / 8;
            const uint bytesPerTrLoad       = bitsPerTransposeLoad(arch, m_elementBits) / 8;
            const uint bytesPerWorkitem     = bytesPerTrLoad * /*numberOfLDSTrLoads*/ 2;
            const uint bytesPerWord         = 4;
            const uint registerCountPerLoad = bytesPerTrLoad / packBytes;
            const uint threadTrLoadOffset   = m_MN * (bytesPerTrLoad + extraLdsBytes);

            auto k = m_context->kernel();

            auto one                = Expression::literal(1);
            auto workitemCountXExpr = Expression::literal(workitemCountX);

            k->setKernelName(toString(elementDataType) + "TransposeLoad");
            k->setKernelDimensions(1);
            k->setWorkgroupSize({workitemCountX, 1, 1});
            k->setWorkitemCount({workitemCountXExpr, one, one});
            m_kernelInvocation.workitemCount = {workitemCountX, 1, 1};
            m_kernelInvocation.workgroupSize = {workitemCountX, 1, 1};

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

            k->addArgument({"a",
                            {DataType::UInt32, PointerType::PointerGlobal},
                            DataDirection::ReadOnly,
                            aExpr});

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
                if(m_elementBits == 6 && m_unalignedVGPRs)
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
                if(m_elementBits == 6)
                {
                    // pad each row so each workitem points to 128b instead of 96b.
                    lds = Register::Value::AllocateLDS(m_context,
                                                       DataType::FP8,
                                                       (m_elementBits * m_MN * m_K) / 8
                                                           + m_MN * (m_K / 16) * extraLdsBytes);
                }
                else
                {
                    lds = Register::Value::AllocateLDS(m_context, elementDataType, m_MN * m_K);
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
                if(m_elementBits == 6)
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
                    co_yield generateOp<Expression::Add>(
                        vLDSPtr, vLDSBasePtr, v256Bits32BytesOffset);
                }
                else
                {
                    co_yield generateOp<Expression::Add>(
                        vLDSPtr, vLDSBasePtr, vLinearWorkitemOffset);
                }

                co_yield m_context->mem()->loadGlobal(vA0, vAPtr, 0, bytesPerTrLoad);
                co_yield m_context->mem()->loadGlobal(vA1, vAPtr, bytesPerTrLoad, bytesPerTrLoad);
                co_yield m_context->mem()
                    ->storeLocal(vLDSPtr, vA0, 0, bytesPerTrLoad)
                    .map(MemoryInstructions::addExtraDst(lds));
                co_yield m_context->mem()
                    ->storeLocal(vLDSPtr, vA1, bytesPerTrLoad + extraLdsBytes, bytesPerTrLoad)
                    .map(MemoryInstructions::addExtraDst(lds));
                co_yield_(m_context->mem()->barrier({lds}));

                co_yield generateOp<Expression::Multiply>(
                    vLinearWordOffset, vWorkitemX, vBytesPerWord);
                co_yield generateOp<Expression::Add>(
                    vTrLoadIdxAddr, vTrLoadIdxAddr, vLinearWordOffset);
                co_yield m_context->mem()
                    ->loadGlobal(vTransposeWorkitemIdx, vTrLoadIdxAddr, 0, bytesPerWord)
                    .map(MemoryInstructions::addExtraSrc(lds));
                if(m_elementBits == 6)
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

                co_yield m_context->mem()
                    ->transposeLoadLocal(
                        vA0T, vLDSPtr, /*offset*/ 0, bytesPerTrLoad + extraLdsBytes, m_elementBits)
                    .map(MemoryInstructions::addExtraSrc(lds));
                co_yield m_context->mem()
                    ->transposeLoadLocal(vA1T,
                                         vLDSPtr,
                                         /*offset*/ threadTrLoadOffset,
                                         bytesPerTrLoad + extraLdsBytes,
                                         m_elementBits)
                    .map(MemoryInstructions::addExtraSrc(lds));

                co_yield_(m_context->mem()->barrier({lds}));
                if(m_elementBits == 6)
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

                    const uint regCount = bitsPerTransposeLoad(arch, m_elementBits) / 32;
                    for(uint regIdx = 0; regIdx < regCount; regIdx++)
                    {
                        co_yield m_context->mem()->storeGlobal(
                            vResultPtr,
                            vA0T->subset({regIdx}),
                            /*offset*/ regIdx * numWorkitemsPerWave * bytesPerWord,
                            bytesPerWord);
                        co_yield m_context->mem()->storeGlobal(
                            vResultPtr,
                            vA1T->subset({regIdx}),
                            /*offset*/ (regCount + regIdx) * numWorkitemsPerWave * bytesPerWord,
                            bytesPerWord);
                    }
                }
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

        void checkGeneratedCode()
        {
            auto const& arch            = m_context->targetArchitecture();
            const uint  dsReadWriteBits = bitsPerTransposeLoad(arch, m_elementBits);
            std::string mnemonic        = transposeLoadMnemonic(arch, m_elementBits);

            std::string globalLoadDWordX{fmt::format("global_load_dwordx{} ", m_dwordX)};
            std::string dsWriteBX{fmt::format("ds_write_b{} ", dsReadWriteBits)};

            if(!m_generated)
            {
                generate();
            }
            std::string code{m_context->instructions()->toString()};

            CHECK(countSubstring(code, mnemonic) == 2);
            CHECK(countSubstring(code, "global_load_dword") == 3);
            CHECK(countSubstring(code, globalLoadDWordX) == 2);

            CHECK(countSubstring(code, "ds_write_b") == 2);
            CHECK(countSubstring(code, dsWriteBX) == 2);
        }

        template <typename StoredAsType>
        void execute()
        {
            auto const& arch = m_context->targetArchitecture();

            std::vector<StoredAsType> data(m_MN * m_K);
            for(int i = 0; i < m_MN; i++)
                for(int j = 0; j < m_K; j++)
                    data[i * m_K + j] = rand() % 10;

            auto packedData = pack<StoredAsType>(data, m_elementBits);
            CHECK(packedData.size() > 0);

            std::vector<uint32_t> result(packedData.size());

            std::vector<uint32_t> trLoadIdx(64);
            {
                const int factor = 2;
                const int NX1    = m_MN / 16;
                const int NX0    = 4 / NX1;
                // each thread points to 64b or 96b
                const int NY0 = bitsPerTransposeLoad(arch, m_elementBits) / m_elementBits;
                const int NY1 = 16 / NY0;
                for(int x0 = 0; x0 < NX0; x0++)
                    for(int x1 = 0; x1 < NX1; x1++)
                        for(int y0 = 0; y0 < NY0; y0++)
                            for(int y1 = 0; y1 < NY1; y1++)
                            {
                                trLoadIdx[NX1 * NY0 * NY1 * x0 + NY0 * NY1 * x1 + NY1 * y0 + y1]
                                    = factor * NX1 * NY0 * NY1 * x0 + NX1 * NY1 * y0 + NY1 * x1
                                      + y1;
                            }
            }

            auto d_a         = make_shared_device(packedData);
            auto d_result    = make_shared_device<uint32_t>(result.size());
            auto d_trLoadIdx = make_shared_device(trLoadIdx);

            this->operator()(m_kernelInvocation, d_result.get(), d_a.get(), d_trLoadIdx.get());

            CHECK_THAT(hipMemcpy(result.data(),
                                 d_result.get(),
                                 sizeof(uint32_t) * result.size(),
                                 hipMemcpyDefault),
                       HasHipSuccess(0));

            auto result_unpacked = unpack<StoredAsType>(result, m_elementBits);
            CHECK(packedData.size() > 0);

            {
                int NY0 = 4;
                int NY2 = 32 / m_elementBits;
                if(m_elementBits == 6)
                {
                    NY0 = 2;
                    NY2 = 96 / m_elementBits;
                }

                const int NY1 = m_K / NY2 / NY0;
                const int NX0 = m_MN / 16;

                for(int y0 = 0; y0 < NY0; y0++)
                    for(int y1 = 0; y1 < NY1; y1++)
                        for(int x0 = 0; x0 < NX0; x0++)
                            for(int x1 = 0; x1 < 4; x1++)
                                for(int x2 = 0; x2 < 4; x2++)
                                    for(int y2 = 0; y2 < NY2; y2++)
                                    {
                                        CHECK(
                                            data[y1 * NY0 * NY2 * NX0 * 16 + y0 * NY2 * NX0 * 16
                                                 + y2 * NX0 * 16 + x0 * 16 + x1 * 4 + x2]
                                            == result_unpacked[y0 * NY1 * NX0 * NY2 * 16
                                                               + y1 * NX0 * NY2 * 16 + x0 * 16 * NY2
                                                               + x1 * 4 * NY2 + x2 * NY2 + y2]);
                                    }
            }
        }

    protected:
        DataType         m_elementType;
        bool             m_unalignedVGPRs;
        uint             m_dwordX;
        uint             m_MN, m_K;
        uint             m_elementBits;
        KernelInvocation m_kernelInvocation;

        template <typename StoredAsType>
        std::vector<uint32_t> pack(const std::vector<StoredAsType>& unpacked, uint elementBits)
        {
            AssertFatal(
                (elementBits == 4 || elementBits == 6 || elementBits == 8 || elementBits == 16)
                && "Only 4, 6, 8, and 16 bits are supported!");

            if(elementBits == 4)
            {
                if constexpr(std::is_same_v<StoredAsType, uint8_t>)
                {
                    return packFP4x8(unpacked);
                }
            }
            else if(elementBits == 6)
            {
                if constexpr(std::is_same_v<StoredAsType, uint8_t>)
                {
                    return packF6x16(unpacked);
                }
            }
            else if(elementBits == 8 || elementBits == 16)
            {
                std::vector<uint32_t> packed(unpacked.size() * sizeof(StoredAsType)
                                             / sizeof(uint32_t));
                std::memcpy(packed.data(), unpacked.data(), sizeof(uint32_t) * packed.size());
                return packed;
            }
            Throw<FatalError>("unreachable");
        }

        template <typename StoredAsType>
        std::vector<StoredAsType> unpack(const std::vector<uint32_t>& packed, uint elementBits)
        {
            AssertFatal(
                (elementBits == 4 || elementBits == 6 || elementBits == 8 || elementBits == 16)
                && "Only 4, 6, 8, and 16 bits are supported!");

            if(elementBits == 4)
            {
                if constexpr(std::is_same_v<StoredAsType, uint8_t>)
                {
                    return unpackFP4x8(packed);
                }
            }
            else if(elementBits == 6)
            {
                if constexpr(std::is_same_v<StoredAsType, uint8_t>)
                {
                    return unpackF6x16(packed);
                }
            }
            else if(elementBits == 8 || elementBits == 16)
            {
                std::vector<StoredAsType> unpacked(packed.size() * sizeof(uint32_t)
                                                   / sizeof(StoredAsType));
                std::memcpy(unpacked.data(), packed.data(), sizeof(StoredAsType) * unpacked.size());
                return unpacked;
            }
            Throw<FatalError>("unreachable");
        }
    };

    TEST_CASE("Assemble transpose load kernel", "[memory-instructions][codegen]")
    {
        auto dataTypeVGPRAligmentDWordX = GENERATE(std::make_tuple(DataType::Half, false, 2),
                                                   std::make_tuple(DataType::BFloat16, false, 2),
                                                   std::make_tuple(DataType::FP8, false, 2),
                                                   std::make_tuple(DataType::BF8, false, 2),
                                                   std::make_tuple(DataType::FP6, true, 3),
                                                   std::make_tuple(DataType::FP6, false, 3),
                                                   std::make_tuple(DataType::BF6, true, 3),
                                                   std::make_tuple(DataType::BF6, false, 3),
                                                   std::make_tuple(DataType::FP4, false, 2));

        auto arch = GENERATE(GPUArchitectureTarget{GPUArchitectureGFX::GFX950},
                             GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.xnack = true}},
                             GPUArchitectureTarget{GPUArchitectureGFX::GFX950, {.sramecc = true}});

        SECTION("For each GFX950 architecture")
        {
            auto context = TestContext::ForTarget(arch);

            SECTION("For each datatype and unalignedVGPRs option")
            {
                auto [type, unalignedVGPRs, dwordX] = dataTypeVGPRAligmentDWordX;

                SECTION("For each wavetile")
                {
                    switch(type)
                    {
                    case DataType::Half:
                    case DataType::BFloat16:
                    {
                        auto [MN, K] = GENERATE(std::make_pair(16, 32), std::make_pair(32, 16));
                        TransposeLoadKernel kernel(
                            context.get(), type, unalignedVGPRs, dwordX, MN, K);
                        kernel.checkGeneratedCode();
                        break;
                    }
                    case DataType::FP8:
                    case DataType::BF8:
                    {
                        auto [MN, K] = GENERATE(std::make_pair(16, 64), std::make_pair(32, 32));
                        TransposeLoadKernel kernel(
                            context.get(), type, unalignedVGPRs, dwordX, MN, K);
                        kernel.checkGeneratedCode();
                        break;
                    }
                    case DataType::FP6:
                    case DataType::BF6:
                    case DataType::FP4:
                    {
                        auto [MN, K] = GENERATE(std::make_pair(16, 128), std::make_pair(32, 64));
                        TransposeLoadKernel kernel(
                            context.get(), type, unalignedVGPRs, dwordX, MN, K);
                        kernel.checkGeneratedCode();
                        break;
                    }
                    default:
                        Throw<FatalError>("unreachable");
                        break;
                    }
                }
            }
        }
    }

    TEST_CASE("Run transpose load kernel", "[memory-instructions][gpu]")
    {
        auto dataTypeVGPRAligmentDWordX = GENERATE(std::make_tuple(DataType::Half, false, 2),
                                                   std::make_tuple(DataType::BFloat16, false, 2),
                                                   std::make_tuple(DataType::FP8, false, 2),
                                                   std::make_tuple(DataType::BF8, false, 2),
                                                   std::make_tuple(DataType::FP6, true, 3),
                                                   std::make_tuple(DataType::FP6, false, 3),
                                                   std::make_tuple(DataType::BF6, true, 3),
                                                   std::make_tuple(DataType::BF6, false, 3),
                                                   std::make_tuple(DataType::FP4, false, 2));

        auto wavetile = GENERATE(std::make_pair(16, 128), std::make_pair(32, 64));
        auto context  = TestContext::ForTestDevice();
        if(context->targetArchitecture().target().gfx != GPUArchitectureGFX::GFX950)
        {
            SKIP("Test only supported on GFX950");
        }

        SECTION("For each datatype and unalignedVGPRs option")
        {
            auto [type, unalignedVGPRs, dwordX] = dataTypeVGPRAligmentDWordX;

            SECTION("For each wavetile")
            {
                switch(type)
                {
                case DataType::Half:
                case DataType::BFloat16:
                {
                    auto [MN, K] = GENERATE(std::make_pair(16, 32), std::make_pair(32, 16));
                    TransposeLoadKernel kernel(context.get(), type, unalignedVGPRs, dwordX, MN, K);
                    kernel.execute<uint16_t>();
                    break;
                }
                case DataType::FP8:
                case DataType::BF8:
                {
                    auto [MN, K] = GENERATE(std::make_pair(16, 64), std::make_pair(32, 32));
                    TransposeLoadKernel kernel(context.get(), type, unalignedVGPRs, dwordX, MN, K);
                    kernel.execute<uint8_t>();
                    break;
                }
                case DataType::FP6:
                case DataType::BF6:
                case DataType::FP4:
                {
                    auto [MN, K] = GENERATE(std::make_pair(16, 128), std::make_pair(32, 64));
                    TransposeLoadKernel kernel(context.get(), type, unalignedVGPRs, dwordX, MN, K);
                    kernel.execute<uint8_t>();
                    break;
                }
                default:
                    Throw<FatalError>("unreachable");
                    break;
                }
            }
        }
    }
}
