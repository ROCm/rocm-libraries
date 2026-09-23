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

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/GPUArchitecture/GPUArchitectureTarget.hpp>
#include <rocRoller/InstructionValues/Register.hpp>
#include <rocRoller/InstructionValues/RegisterAllocator_detail.hpp>
#include <rocRoller/KernelOptions_detail.hpp>
#include <rocRoller/Scheduling/Observers/VGPRIndexingObserver_detail.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include "catch2/generators/catch_generators_range.hpp"
#include "common/Utilities.hpp"

#include "TestContext.hpp"

using namespace rocRoller;

namespace VGPRIndexingObserverTests
{
    class VGPRIndexingObserverTest : public TestContext
    {
    public:
        VGPRIndexingObserverTest(GPUArchitectureTarget target)
            : TestContext(TestContext::ForTarget(target)){};
    };

    using namespace Register::RegisterAllocatorDetail;
    using namespace Scheduling::VGPRIndexingObserverDetail;

    static const std::string setModeStr = "s_set_vgpr_msb";
    static const std::string wmmaStr    = "v_wmma_bf16_16x16x32_bf16";

    TEST_CASE("VGPR Indexing - Test all bank combinations", "[codegen]")
    {
        auto target = GENERATE(GPUArchTargetGFX1250Rev0, GPUArchTargetGFX1250Rev1);

        VGPRIndexingObserverTest t{target};

        auto vBase = t.createRegisters(
            Register::Type::Vector, DataType::Float, 1024 - ReservedRegionSize(), 1);

        auto v = [&](auto idx) {
            AssertFatal(idx >= ReservedRegionSize());
            return vBase[idx - ReservedRegionSize()];
        };

        auto generateCombinationArrays = []() {
            auto numBankVGPRCombos = 256; // 8-bit mode register, 2^8 combinations

            std::vector<std::array<int, 4>> combos;
            combos.reserve(numBankVGPRCombos);
            for(auto i = 0; i < numBankVGPRCombos; i++)
            {
                combos.push_back({(i / 64) % 4, (i / 16) % 4, (i / 4) % 4, (i / 1) % 4});
            }
            return combos;
        };

        static const auto bankCombos = generateCombinationArrays();

        auto primer = Instruction(setModeStr, {Register::Value::Literal(0)}, {}, {}, "");
        t->schedule(primer);

        auto banks  = GENERATE_COPY(from_range(bankCombos));
        auto getIdx = [&](auto i) { return banks[i] * 256 + ReservedRegionSize() + i; };

        std::string banksSectionDescriptor = "Bank Combo: (";
        for(auto i = 0; i < banks.size(); i++)
        {
            banksSectionDescriptor += std::to_string(banks[i]);
            if(i + 1 < banks.size())
                banksSectionDescriptor += ", ";
        }
        banksSectionDescriptor += ")";

        DYNAMIC_SECTION(banksSectionDescriptor)
        {
            auto v0Idx = getIdx(0); // dest
            auto v1Idx = getIdx(1); // src0
            auto v2Idx = getIdx(2); // src1
            auto v3Idx = getIdx(3); // src2
            int  expectedMode
                = static_cast<int>((GetBankBits(v0Idx) << 6) | (GetBankBits(v3Idx) << 4)
                                   | (GetBankBits(v2Idx) << 2) | (GetBankBits(v1Idx)));

            auto expectedSet  = setModeStr + " " + std::to_string(expectedMode);
            auto expectedWMMA = wmmaStr + " "
                                + fmt::format("v{}, v{}, v{}, v{}",
                                              ReservedRegionSize(),
                                              ReservedRegionSize() + 1,
                                              ReservedRegionSize() + 2,
                                              ReservedRegionSize() + 3);

            auto inst = Instruction(wmmaStr, {v(v0Idx)}, {v(v1Idx), v(v2Idx), v(v3Idx)}, {}, "");
            t->schedule(inst);

            CHECK(1 == countSubstring(t.output(), expectedSet));
            CHECK(1 == countSubstring(t.output(), expectedWMMA));

            t->instructions()->clear();
        }
    }

    TEST_CASE("VGPR Indexing - Test duplicate mode set", "[codegen]")
    {
        auto target = GENERATE(GPUArchTargetGFX1250Rev0, GPUArchTargetGFX1250Rev1);
        VGPRIndexingObserverTest t{target};

        auto vBase = t.createRegisters(
            Register::Type::Vector, DataType::Float, 1024 - ReservedRegionSize(), 1);

        auto v = [&](auto idx) {
            AssertFatal(idx >= ReservedRegionSize());
            return vBase[idx - ReservedRegionSize()];
        };

        auto expectedWMMA = wmmaStr + " v0, v1, v2, v3";

        SECTION("Same mode: expect no duplicate set")
        {
            std::vector<Instruction> insts
                = {Instruction(wmmaStr, {v(256)}, {v(257), v(514), v(771)}, {}, ""),
                   Instruction(wmmaStr, {v(256)}, {v(257), v(514), v(771)}, {}, "")};

            t->schedule(insts[0]);
            CHECK(1 == countSubstring(t.output(), setModeStr));
            CHECK(1 == countSubstring(t.output(), expectedWMMA));

            t->schedule(insts[1]);
            CHECK(1 == countSubstring(t.output(), setModeStr));
            CHECK(2 == countSubstring(t.output(), expectedWMMA));
        }

        SECTION("Different mode: expect separate set")
        {
            std::vector<Instruction> insts
                = {Instruction(wmmaStr, {v(256)}, {v(257), v(514), v(771)}, {}, ""),
                   Instruction(wmmaStr, {v(512)}, {v(257), v(514), v(771)}, {}, "")};

            t->schedule(insts[0]);
            CHECK(1 == countSubstring(t.output(), setModeStr));
            CHECK(1 == countSubstring(t.output(), expectedWMMA));

            t->schedule(insts[1]);
            CHECK(2 == countSubstring(t.output(), setModeStr));
            CHECK(2 == countSubstring(t.output(), expectedWMMA));
        }
    }

    TEST_CASE("VGPR Indexing - Test MODE set skips non-vector operands", "[codegen]")
    {
        auto target = GENERATE(GPUArchTargetGFX1250Rev0, GPUArchTargetGFX1250Rev1);
        VGPRIndexingObserverTest t{target};

        auto vBase = t.createRegisters(
            Register::Type::Vector, DataType::Float, 1024 - ReservedRegionSize(), 1);

        auto v = [&](auto idx) {
            AssertFatal(idx >= ReservedRegionSize());
            return vBase[idx - ReservedRegionSize()];
        };

        int dstIdx  = 256; // dest
        int src1Idx = 257; // src1

        auto src0Literal = Register::Value::Literal(4); // src0

        int expectedMode
            = static_cast<int>(0 | (GetBankBits(dstIdx) << 6) | (GetBankBits(src1Idx) << 2));

        auto expectedSet = setModeStr + " " + std::to_string(expectedMode);

        auto inst = Instruction("v_lshrrev_b32", {v(dstIdx)}, {src0Literal, v(src1Idx)}, {}, "");
        t->schedule(inst);

        CHECK(1 == countSubstring(t.output(), expectedSet));
        CHECK(1 == countSubstring(t.output(), "v_lshrrev_b32 v0, 4, v1"));

        t->instructions()->clear();
    }
}
