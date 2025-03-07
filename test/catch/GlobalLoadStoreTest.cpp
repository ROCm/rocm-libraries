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

#include <cmath>
#include <memory>

#include "CustomMatchers.hpp"
#include "CustomSections.hpp"
#include "TestContext.hpp"
#include "TestKernels.hpp"

#include <common/SourceMatcher.hpp>
#include <common/TestValues.hpp>

#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/Instruction.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>

#include <catch2/catch_test_macros.hpp>

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace rocRoller;

namespace GlobalLoadStoreInstructionsTest
{
    struct GlobalLoadStoreKernel : public AssemblyTestKernel
    {
        GlobalLoadStoreKernel(ContextPtr context, size_t numBytes)
            : AssemblyTestKernel(context)
            , m_numBytes(numBytes)

        {
            auto arch = m_context->targetArchitecture().target();
            if(!arch.isCDNAGPU())
                SKIP("Test not yet supported on " << arch);
        }

        void generate() override
        {
            auto k = m_context->kernel();

            k->setKernelName("GlobalLoadStoreTest");
            k->setKernelDimensions(1);

            k->addArgument({"result",
                            {DataType::UInt32, PointerType::PointerGlobal},
                            DataDirection::WriteOnly});
            k->addArgument(
                {"data", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::ReadOnly});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_result, s_data;
                co_yield m_context->argLoader()->getValue("result", s_result);
                co_yield m_context->argLoader()->getValue("data", s_data);

                auto const registerCount = 1;

                auto v_result
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::UInt32, PointerType::PointerGlobal},
                                                   registerCount);

                auto v_data = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::Int64, registerCount);

                auto v_addr
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   DataType::UInt32,
                                                   registerCount,
                                                   Register::AllocationOptions::FullyContiguous());

                co_yield v_data->allocate();
                co_yield v_addr->allocate();
                co_yield v_result->allocate();

                co_yield m_context->copier()->copy(v_result, s_result, "Move result pointer.");
                co_yield m_context->copier()->copy(v_data, s_data, "Move input data pointer.");

                for(int i = 0; i < m_numBytes / 4; i++)
                {
                    // Each iteration writes and reads 4 bytes (one register)
                    const auto offset = i * 4;
                    co_yield m_context->mem()->loadGlobal(v_addr, v_data, offset, 4);
                    co_yield m_context->mem()->storeGlobal(v_result, v_addr, offset, 4);
                    co_yield m_context->mem()->barrier();
                }
            };

            m_context->schedule(kb());
            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

    protected:
        size_t m_numBytes;
    };

    TEST_CASE("Run global load/store kernel", "[global-load-store][gpu]")
    {
        auto context = TestContext::ForTestDevice();

        std::vector<int> data((1 << 13)); // 8192 exceeds maxOffset
        std::iota(data.begin(), data.end(), 1);

        GlobalLoadStoreKernel kernel(context.get(), data.size() * sizeof(int));

        auto d_data   = make_shared_device(data);
        auto d_result = make_shared_device<int>(data.size());

        kernel({}, d_result.get(), d_data);

        REQUIRE_THAT(d_result, HasDeviceVectorEqualTo(data));
    }

}
