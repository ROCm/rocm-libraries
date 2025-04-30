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

namespace largerLDSTest
{
    struct LargerLDSKernel : public AssemblyTestKernel
    {
        LargerLDSKernel(rocRoller::ContextPtr context, int numBytes, int numLoads)
            : AssemblyTestKernel(context)
            , m_numBytes(numBytes)
            , m_numLoads(numLoads)
        {
            auto const& arch = m_context->targetArchitecture().target();
            if(!arch.isCDNA35GPU())
                SKIP("Test not yet supported on " << arch);

            auto maxLDS  = context->targetArchitecture().GetCapability(GPUCapability::MaxLdsSize);
            auto ldsSize = m_numBytes * m_numLoads;
            if(ldsSize > maxLDS)
                SKIP("LDS size " << ldsSize << " exceeds maxLDS " << maxLDS);
        }

        void generate() override
        {
            auto k = m_context->kernel();

            k->setKernelName("LargerLDSKernel");
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

                int size = (m_numBytes % 4 == 0) ? m_numBytes / 4 : m_numBytes / 4 + 1;

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::UInt32, PointerType::PointerGlobal},
                                                   1);
                auto v_ptr
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   DataType::UInt32,
                                                   size,
                                                   Register::AllocationOptions::FullyContiguous());

                auto v_a = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, 1);

                auto lds = Register::Value::AllocateLDS(
                    m_context, DataType::UInt32, (m_numBytes * m_numLoads) / 4);
                auto lds_offset = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);

                co_yield v_a->allocate();
                co_yield v_ptr->allocate();
                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move result pointer.");
                co_yield m_context->copier()->copy(v_a, s_a, "Move input data pointer.");
                co_yield m_context->copier()->copy(
                    lds_offset,
                    Register::Value::Literal(lds->getLDSAllocation()->offset()),
                    "Move lds offset");

                for(int i = 0; i < m_numLoads; i++)
                {

                    const auto offset = i * m_numBytes;
                    co_yield m_context->mem()->loadGlobal(v_ptr, v_a, offset, m_numBytes);
                    co_yield m_context->mem()->storeLocal(lds_offset, v_ptr, offset, m_numBytes);
                    co_yield m_context->mem()->barrier();
                    co_yield m_context->mem()->loadLocal(v_ptr, lds_offset, offset, m_numBytes);
                    co_yield m_context->mem()->storeGlobal(v_result, v_ptr, offset, m_numBytes);
                }
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

    protected:
        int m_numBytes, m_numLoads;
    };

    TEST_CASE("Assemble larger lds test 1", "[largerLDS][codegen]")
    {
        SUPPORTED_ARCH_SECTION(arch)
        {
            auto      context  = TestContext::ForTarget(arch);
            const int numBytes = 256;
            const int numLoads = 512;

            LargerLDSKernel kernel(context.get(), numBytes, numLoads);
            CHECK(kernel.getAssembledKernel().size() > 0);
        }
    }

    TEST_CASE("Run larger lds test", "[largerLDS][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        const int numBytes = 256;
        const int numLoads = 512;

        LargerLDSKernel kernel(context.get(), numBytes, numLoads);

        std::vector<int> a(numBytes * numLoads / 4);
        for(int i = 0; i < a.size(); i++)
            a[i] = i + 11;

        auto d_a      = make_shared_device(a);
        auto d_result = make_shared_device<int>(a.size());

        kernel({}, d_result.get(), d_a);

        REQUIRE_THAT(d_result, HasDeviceVectorEqualTo(a));
    }
}
