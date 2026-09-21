// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "Agent.hpp"

#include <common/SourceMatcher.hpp>
#include <common/Utilities.hpp>
#include <rocRoller/AssemblyKernel.hpp>
#include <rocRoller/CodeGen/ArgumentLoader.hpp>
#include <rocRoller/CodeGen/CopyGenerator.hpp>
#include <rocRoller/CodeGen/MemoryInstructions.hpp>
#include <rocRoller/CommandSolution.hpp>
#include <rocRoller/Context.hpp>
#include <rocRoller/ExecutableKernel.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureLibrary.hpp>
#include <rocRoller/KernelArguments.hpp>
#include <rocRoller/Operations/Command.hpp>
#include <rocRoller/Operations/CommandArgument.hpp>
#include <rocRoller/Utilities/Generator.hpp>
#include <rocRoller/Utilities/HipUtils.hpp>

#include "../catch/CustomMatchers.hpp"
#include "../catch/TestContext.hpp"
#include "../catch/TestKernels.hpp"

#include <fmt/format.h>

#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>

#include "hip/hip_runtime.h"

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <ranges>
#include <thread>
#include <vector>

using namespace rocRoller;

namespace RocprofilerTest
{
    class RocprofilerTestKernel : public AssemblyTestKernel
    {
        KernelInvocation m_invocation;

    public:
        RocprofilerTestKernel(ContextPtr context, KernelInvocation invocation)
            : AssemblyTestKernel(context)
            , m_invocation(invocation)
        {
            auto k = m_context->kernel();
            k->setKernelDimensions(1);
            k->setWorkgroupSize(m_invocation.workgroupSize);

            k->setWorkitemCount(
                {std::make_shared<Expression::Expression>(m_invocation.workitemCount[0]),
                 std::make_shared<Expression::Expression>(m_invocation.workitemCount[1]),
                 std::make_shared<Expression::Expression>(m_invocation.workitemCount[2])});
        }

        template <typename... Args>
        void operator()(Args const&... args)
        {
            AssemblyTestKernel::operator()(m_invocation, args...);
        }
    };

    class AddTestKernel : public RocprofilerTestKernel
    {
    public:
        AddTestKernel(ContextPtr context,
                      uint32_t   literal,
                      uint       workgroupSize = 256,
                      uint       workitemCount = 256 * 256)
            : RocprofilerTestKernel(
                context, KernelInvocation{{workitemCount, 1, 1}, {workgroupSize, 1, 1}, 0})
            , m_literal(literal)
        {
        }

    protected:
        void generate() override
        {
            auto k = m_context->kernel();

            k->addArgument(
                {"ptr", {DataType::UInt32, PointerType::PointerGlobal}, DataDirection::WriteOnly});
            k->addArgument({"val", {DataType::UInt32}, DataDirection::ReadOnly});

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                Register::ValuePtr s_ptr, s_value;
                co_yield m_context->argLoader()->getValue("ptr", s_ptr);
                co_yield m_context->argLoader()->getValue("val", s_value);

                auto v_ptr
                    = Register::Value::Placeholder(m_context,
                                                   Register::Type::Vector,
                                                   {DataType::UInt32, PointerType::PointerGlobal},
                                                   1);
                auto v_value = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);

                co_yield v_ptr->allocate();
                co_yield m_context->copier()->copy(v_ptr, s_ptr, "Move pointer");

                co_yield v_value->allocate();
                co_yield Expression::generate(v_value, Expression::literal(m_literal), m_context);
                co_yield Expression::generate(
                    v_value, v_value->expression() + s_value->expression(), m_context);

                co_yield m_context->mem()->storeGlobal(v_ptr, v_value, 0, 4);
            };

            m_context->schedule(kb());

            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

    private:
        uint32_t m_literal;
    };

    class MovTestKernel : public RocprofilerTestKernel
    {
    public:
        MovTestKernel(ContextPtr context,
                      uint32_t   literal,
                      uint       workgroupSize = 256,
                      uint       workitemCount = 256 * 256)
            : RocprofilerTestKernel(
                context, KernelInvocation{{workitemCount, 1, 1}, {workgroupSize, 1, 1}, 0})
            , m_literal(literal)
        {
        }

    protected:
        void generate() override
        {
            auto k = m_context->kernel();

            m_context->schedule(k->preamble());
            m_context->schedule(k->prolog());

            auto kb = [&]() -> Generator<Instruction> {
                auto v_value = Register::Value::Placeholder(
                    m_context, Register::Type::Vector, DataType::UInt32, 1);

                co_yield v_value->allocate();
                co_yield Expression::generate(v_value, Expression::literal(m_literal), m_context);
            };

            m_context->schedule(kb());

            m_context->schedule(k->postamble());
            m_context->schedule(k->amdgpu_metadata());
        }

    private:
        uint32_t m_literal;
    };

    TEST_CASE("Rocprofiler add kernel", "[rocprofiler][gpu]")
    {
        rocRoller::profiler::reset();

        auto literal    = GENERATE(0xdeadbeef, 0x12345678, 0xabcdef00);
        auto commandArg = GENERATE(7, 21, 331);

        std::string const testName = fmt::format("add_0x{:x}_value_{}", literal, commandArg);

        auto testContext = TestContext::ForTestDevice({}, testName);
        auto context     = testContext.get();

        AddTestKernel kernel(context, literal);
        auto          d_ptr = make_shared_device<uint32_t>(1, 0);

        const auto latencies = rocRoller::profiler::loopUntilDispatchData(
            [&]() { kernel(d_ptr.get(), commandArg); });

        CHECK_THAT(d_ptr, HasDeviceScalarEqualTo(commandArg + literal));

        std::stringstream ss;
        ss << "Instruction, Total Latency, Hit Count, Average Latency" << std::endl;
        for(const auto& data : latencies)
        {
            uint64_t avg_latency = data.meanLatency();
            ss << "\"" << data.instruction << "\", " << data.totalLatency << ", " << data.hitcount
               << ", " << avg_latency << std::endl;
        }
        INFO(ss.str());

        {
            const auto target = context->targetArchitecture().target();
            if(target.isCDNA1GPU() || target.isRDNAGPU())
                REQUIRE(latencies.size() == 9);
            else
                REQUIRE(latencies.size() == 8);
        }

        { // Ensure instructions exist in expected quantities in the profile data
            std::string const instructionsStr = [&]() {
                std::stringstream ss;
                streamJoin(
                    ss,
                    std::views::transform(latencies, [](const auto& d) { return d.instruction; }),
                    "\n");
                return ss.str();
            }();
            INFO("Instructions:\n" << instructionsStr);
            CHECK(1
                  == countSubstring(instructionsStr,
                                    fmt::format("v_mov_b32_e32 v1, 0x{:x}", literal)));
        }
    }

    TEST_CASE("Rocprofiler different literals in assembly", "[rocprofiler][gpu]")
    {
        /*
        Ensure callbacks are properly handled and correctly mapped to a dispatch.
        Kernels with different literals are launched in various orders.
        Ensures the profiler returns instructions from the correct kernel.
        */
        rocRoller::profiler::reset();

        std::vector<uint32_t> literals
            = {0xbeef0000, 0xbeef0001, 0xbeef0002, 0xbeef0003, 0xbeef0004, 0xbeef0005, 0xbeef0006};

        std::vector<size_t> order;
        std::string         sectionName;

        SECTION("Order 1")
        {
            order       = {0, 1, 2, 1};
            sectionName = "order_1";
        }

        SECTION("Order 2")
        {
            order       = {3, 4};
            sectionName = "order_2";
        }

        SECTION("Order 3")
        {
            order       = {6, 5, 4, 3, 2, 1, 0};
            sectionName = "order_3";
        }

        INFO(sectionName);

        std::vector<MovTestKernel> kernels;
        kernels.reserve(literals.size());

        for(size_t i = 0; i < literals.size(); ++i)
        {
            std::string const testName
                = fmt::format("{}_literal_0x{:x}_kernel_{}", sectionName, literals[i], i);
            auto testContext = TestContext::ForTestDevice({}, testName);
            auto context     = testContext.get();
            kernels.emplace_back(context, literals[i]);
        }

        for(size_t i = 0; i < order.size() - 1; ++i)
        {
            kernels[order[i]]();
        }

        auto latencies
            = rocRoller::profiler::loopUntilDispatchData([&]() { kernels[order.back()](); });

        auto literalHex = fmt::format("0x{:x}", literals[order.back()]);

        CAPTURE(literalHex);
        INFO(toString(latencies));
        REQUIRE(latencies.size() == 2);
        CHECK(1 == countSubstring(latencies[0].instruction, literalHex));
        CHECK(latencies[1].instruction == "s_endpgm");
    }

    TEST_CASE("Rocprofiler two kernels in loopUntilDispatchData", "[rocprofiler][gpu]")
    {
        rocRoller::profiler::reset();

        const auto literal1 = 0xbeef1111;
        const auto literal2 = 0xbeef2222;

        auto          testContext1 = TestContext::ForTestDevice({}, "1");
        auto          context1     = testContext1.get();
        MovTestKernel kernel1(context1, literal1);

        auto          testContext2 = TestContext::ForTestDevice({}, "2");
        auto          context2     = testContext2.get();
        MovTestKernel kernel2(context2, literal2);

        const auto latencies = rocRoller::profiler::loopUntilDispatchData([&]() {
            kernel1();
            kernel2();
            // Current implementation captures first dispatch
        });

        const auto literal1Hex = fmt::format("0x{:x}", literal1);
        const auto literal2Hex = fmt::format("0x{:x}", literal2);

        CAPTURE(literal1Hex, literal2Hex);
        INFO(toString(latencies));
        REQUIRE(latencies.size() == 2);
        CHECK(1 == countSubstring(latencies[0].instruction, literal1Hex));
        CHECK(0 == countSubstring(latencies[0].instruction, literal2Hex));
        CHECK(latencies[1].instruction == "s_endpgm");
    }

    TEST_CASE("Rocprofiler small workgroup count", "[rocprofiler][gpu]")
    {
        /*
        With a small workgroup count, the filtered-for SE/CU/SIMD may not be used.
        Ensure looping the dispatch works as expected.
        */
        rocRoller::profiler::reset();

        const auto literal = 0xdead1234;

        auto testContext = TestContext::ForTestDevice({}, "small_workgroup_count");
        auto context     = testContext.get();

        MovTestKernel kernel(context, literal, 64, 64 * 64);

        const auto latencies = rocRoller::profiler::loopUntilDispatchData([&]() { kernel(); });

        const auto literalHex = fmt::format("0x{:x}", literal);

        CAPTURE(literalHex);
        INFO(toString(latencies));
        REQUIRE(latencies.size() == 2);
        CHECK(1 == countSubstring(latencies[0].instruction, literalHex));
        CHECK(latencies[1].instruction == "s_endpgm");
    }
} // namespace RocprofilerTest
