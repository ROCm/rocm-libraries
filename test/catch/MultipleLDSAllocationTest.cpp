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

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <common/Utilities.hpp>
#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>

using namespace rocRoller;

namespace multipleLDSAllocTest
{
    struct MultipleLDSAllocKernel : public AssemblyTestKernel
    {
        MultipleLDSAllocKernel(rocRoller::ContextPtr context,
                               int                   numBytes1,
                               int                   numLoads1,
                               int                   numBytes2,
                               int                   numLoads2,
                               int                   numBytes3,
                               int                   numLoads3)
            : AssemblyTestKernel(context)
            , m_numBytes1(numBytes1)
            , m_numLoads1(numLoads1)
            , m_numBytes2(numBytes2)
            , m_numLoads2(numLoads2)
            , m_numBytes3(numBytes3)
            , m_numLoads3(numLoads3)
        {
            auto const& arch = m_context->targetArchitecture().target();
            if(!arch.isCDNA35GPU())
                SKIP("Test not yet supported on " << arch);

            auto maxLDS = context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize);
            auto ldsSize
                = m_numBytes1 * m_numLoads1 + m_numBytes2 * m_numLoads2 + m_numBytes3 * m_numLoads3;
            if(ldsSize > maxLDS)
                SKIP("LDS size " << ldsSize << " exceeds maxLDS " << maxLDS);
        }

        void generate() override
        {
            auto k = m_context->kernel();

            k->setKernelName("MultipleLDSAllocKernel");
            k->setKernelDimensions(1);

            k->addArgument({"result",
                            {DataType::UInt32, PointerType::PointerGlobal},
                            DataDirection::WriteOnly});
            k->addArgument(
                {"a", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::ReadOnly});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_a;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("a", s_a);

                auto lds1 = Register::Value::AllocateLDS(
                    m_context, DataType::UInt32, (m_numBytes1 * m_numLoads1) / 4);
                auto lds2 = Register::Value::AllocateLDS(
                    m_context, DataType::UInt32, (m_numBytes2 * m_numLoads2) / 4);
                auto lds3 = Register::Value::AllocateLDS(
                    m_context, DataType::UInt32, (m_numBytes3 * m_numLoads3) / 4);

                int size1 = (m_numBytes1 % 4 == 0) ? m_numBytes1 / 4 : m_numBytes1 / 4 + 1;
                int size2 = (m_numBytes2 % 4 == 0) ? m_numBytes2 / 4 : m_numBytes2 / 4 + 1;
                int size3 = (m_numBytes1 % 4 == 0) ? m_numBytes3 / 4 : m_numBytes3 / 4 + 1;

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::UInt32, PointerType::PointerGlobal},
                                                   1);

                auto v_a = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);

                auto lds1_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                auto lds2_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                auto lds3_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int32, 1);

                co_yield v_a->allocate();
                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move result pointer.");
                co_yield m_context->copier()->copy(v_a, s_a, "Move input data pointer.");
                co_yield m_context->copier()->copy(
                    lds1_offset,
                    Register::Value::Literal(lds1->getLDSAllocation()->offset()),
                    "Move lds1 offset pointer");
                co_yield m_context->copier()->copy(
                    lds2_offset,
                    Register::Value::Literal(lds2->getLDSAllocation()->offset()),
                    "Move lds2 offset pointer");
                co_yield m_context->copier()->copy(
                    lds3_offset,
                    Register::Value::Literal(lds3->getLDSAllocation()->offset()),
                    "Move lds3 offset pointer");

                for(int i = 0; i < m_numLoads1; i++)
                {
                    auto v_ptr = Register::Value::Placeholder(
                        m_context,
                        Register::Type::Vector,
                        DataType::UInt32,
                        size1,
                        Register::AllocationOptions::FullyContiguous());
                    co_yield v_ptr->allocate();
                    const auto offset = i * m_numBytes1;
                    co_yield m_context->mem()->loadGlobal(v_ptr, v_a, offset, m_numBytes1);

                    co_yield m_context->mem()
                        ->storeLocal(lds1_offset, v_ptr, offset, m_numBytes1)
                        .map(MemoryInstructions::addExtraDst(lds1));

                    co_yield_(m_context->mem()->barrier({lds1}));

                    co_yield m_context->mem()
                        ->loadLocal(v_ptr, lds1_offset, offset, m_numBytes1)
                        .map(MemoryInstructions::addExtraSrc(lds1));

                    co_yield m_context->mem()->storeGlobal(v_result, v_ptr, offset, m_numBytes1);
                }
                auto usedLDS = m_numBytes1 * m_numLoads1;

                for(int i = 0; i < m_numLoads2; i++)
                {
                    auto v_ptr = Register::Value::Placeholder(
                        m_context,
                        Register::Type::Vector,
                        DataType::UInt32,
                        size2,
                        Register::AllocationOptions::FullyContiguous());
                    co_yield v_ptr->allocate();
                    const auto offset = i * m_numBytes2;
                    co_yield m_context->mem()->loadGlobal(

                        v_ptr, v_a, usedLDS + offset, m_numBytes2);
                    co_yield m_context->mem()
                        ->storeLocal(lds2_offset, v_ptr, offset, m_numBytes2)
                        .map(MemoryInstructions::addExtraDst(lds2));

                    co_yield_(m_context->mem()->barrier({lds2}));

                    co_yield m_context->mem()
                        ->loadLocal(v_ptr, lds2_offset, offset, m_numBytes2)
                        .map(MemoryInstructions::addExtraSrc(lds2));

                    co_yield m_context->mem()->storeGlobal(
                        v_result, v_ptr, usedLDS + offset, m_numBytes2);
                }
                usedLDS += m_numBytes2 * m_numLoads2;

                for(int i = 0; i < m_numLoads3; i++)
                {
                    auto v_ptr = Register::Value::Placeholder(
                        m_context,
                        Register::Type::Vector,
                        DataType::UInt32,
                        size3,
                        Register::AllocationOptions::FullyContiguous());
                    co_yield v_ptr->allocate();
                    const auto offset = i * m_numBytes3;
                    co_yield m_context->mem()->loadGlobal(
                        v_ptr, v_a, usedLDS + offset, m_numBytes3);

                    co_yield m_context->mem()
                        ->storeLocal(lds3_offset, v_ptr, offset, m_numBytes3)
                        .map(MemoryInstructions::addExtraDst(lds3));

                    co_yield_(m_context->mem()->barrier({lds3}));

                    co_yield m_context->mem()
                        ->loadLocal(v_ptr, lds3_offset, offset, m_numBytes3)
                        .map(MemoryInstructions::addExtraDst(lds3));

                    co_yield m_context->mem()->storeGlobal(
                        v_result, v_ptr, usedLDS + offset, m_numBytes3);
                }
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

    protected:
        int m_numBytes1, m_numLoads1, m_numBytes2, m_numLoads2, m_numBytes3, m_numLoads3;
    };

    TEST_CASE("Assemble multiple lds allocation test", "[lds][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto       context   = TestContext::ForTarget(arch);
            const auto numBytes1 = 256;
            const auto numLoads1 = 70;
            const auto numBytes2 = 128;
            const auto numLoads2 = 128;
            const auto numBytes3 = 64;
            const auto numLoads3 = 512;

            MultipleLDSAllocKernel kernel(
                context.get(), numBytes1, numLoads1, numBytes2, numLoads2, numBytes3, numLoads3);
            CHECK(kernel.getAssembledKernel().size() > 0);
        }
    }

    TEST_CASE("Run multiple lds allocation test", "[lds][codegen][gpu]")
    {
        auto       context   = TestContext::ForTestDevice();
        const auto numBytes1 = 256;
        const auto numLoads1 = 70;
        const auto numBytes2 = 128;
        const auto numLoads2 = 128;
        const auto numBytes3 = 64;
        const auto numLoads3 = 512;

        MultipleLDSAllocKernel kernel(
            context.get(), numBytes1, numLoads1, numBytes2, numLoads2, numBytes3, numLoads3);

        std::vector<int> a((numBytes1 * numLoads1 + numBytes2 * numLoads2 + numBytes3 * numLoads3)
                           / 4);
        std::cout << "total size: " << a.size();
        for(int i = 0; i < a.size(); i++)
            a[i] = i + 11;

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<int>(a.size());

        kernel({}, d_result.get(), d_a);

        REQUIRE_THAT(d_result, HasDeviceVectorEqualTo(a));
    }
}
