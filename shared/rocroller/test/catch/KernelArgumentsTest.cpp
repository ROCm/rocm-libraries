/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2017-2025 AMD ROCm(TM) Software
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

#include <rocRoller/KernelArguments.hpp>

#include <cstddef>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace rocRoller;

TEST_CASE("KernelArguments simple test", "[kernel-arguments]")
{
    KernelArguments args(true);

    std::vector<float> array(5);

    struct
    {
        void*       d;
        const void* c;
        const void* a;
        const void* b;

        int    x;
        double y;
        float  z;
        char   t;

        size_t k;

    } ref;

    // Padding bytes would be left uninitialized in the struct but will be
    // zero-filled by the KernelArguments class. Set them to zero to prevent a
    // test failure. The cast to void* is to avoid the warning about potential
    // invalid floating point configurations.
    memset((void*)&ref, 0, sizeof(ref));

    ref.d = array.data();
    ref.c = array.data() + 1;
    ref.a = array.data() + 2;
    ref.b = array.data() + 3;
    ref.x = 23;
    ref.y = 90.2;
    ref.z = 16.0f;
    ref.t = 'w';
    ref.k = std::numeric_limits<size_t>::max();

    args.append("d", ref.d);
    args.append("c", ref.c);
    args.append("a", ref.a);
    args.append("b", ref.b);

    args.append("x", ref.x);
    args.append("y", ref.y);
    args.append("z", ref.z);
    args.append("t", ref.t);
    args.append("k", ref.k);

    CHECK(args.size() == sizeof(ref));

    std::vector<uint8_t> reference(sizeof(ref), 0);
    memcpy(reference.data(), &ref, sizeof(ref));

    std::vector<uint8_t> result(args.size());
    memcpy(result.data(), args.data(), args.size());

    CHECK(result.size() == reference.size());
    for(size_t i = 0; i < std::min(result.size(), reference.size()); i++)
    {
        CAPTURE(i);
        CHECK(static_cast<uint32_t>(result[i]) == static_cast<uint32_t>(reference[i]));
    }
}

TEST_CASE("KernelArguments binding test", "[kernel-arguments]")
{
    KernelArguments args(true);

    std::vector<float> array(5);

    struct
    {
        void*       d;
        const void* c;
        const void* a;
        const void* b;

        int    x;
        double y;
        float  z;
        char   t;

        size_t k;

    } ref;

    // Padding bytes would be left uninitialized in the struct but will be
    // zero-filled by the KernelArguments class. Set them to zero to prevent a
    // test failure. The cast to void* is to avoid the warning about potential
    // invalid floating point configurations.
    memset((void*)&ref, 0, sizeof(ref));

    ref.d = array.data();
    ref.c = array.data() + 1;
    ref.a = array.data() + 2;
    ref.b = array.data() + 3;
    ref.x = 23;
    ref.y = 90.2;
    ref.z = 16.0f;
    ref.t = 'w';
    ref.k = std::numeric_limits<size_t>::max();

    args.append("d", ref.d);
    args.append("c", ref.c);
    args.append("a", ref.a);
    args.append("b", ref.b);

    args.appendUnbound<int>("x");
    args.append("y", ref.y);
    args.append("z", ref.z);
    args.append("t", ref.t);
    args.append("k", ref.k);

    CHECK(args.size() == sizeof(ref));

    // std::cout << args << std::endl;

    CHECK_THROWS_AS(args.data(), std::runtime_error);

    args.bind("x", ref.x);

    std::vector<uint8_t> reference(sizeof(ref), 0);
    memcpy(reference.data(), &ref, sizeof(ref));

    std::vector<uint8_t> result(args.size());
    memcpy(result.data(), args.data(), args.size());

    CHECK(result.size() == reference.size());
    for(size_t i = 0; i < std::min(result.size(), reference.size()); i++)
    {
        CAPTURE(i);
        CHECK(static_cast<uint32_t>(result[i]) == static_cast<uint32_t>(reference[i]));
    }

    // std::cout << args << std::endl;
}

TEST_CASE("KernelArguments iterator test", "[kernel-arguments]")
{
    // Quick test for required m_log
    {
        KernelArguments args(false);
        CHECK_THROWS_AS(args.begin(), std::runtime_error);
    }

    KernelArguments args(true);

    std::vector<float> array(5);

    struct RefT
    {
        void*       d;
        const void* c;
        const void* a;
        const void* b;

        int    x;
        double y;
        float  z;
        char   t;

        size_t k;

        bool operator==(RefT const& other) const
        {
            return d == other.d && c == other.c && a == other.a && b == other.b && x == other.x
                   && y == other.y && z == other.z && t == other.t && k == other.k;
        }

    } ref;

    ref.d = array.data();
    ref.c = array.data() + 1;
    ref.a = array.data() + 2;
    ref.b = array.data() + 3;
    ref.x = 23;
    ref.y = 90.2;
    ref.z = 16.0f;
    ref.t = 'w';
    ref.k = std::numeric_limits<size_t>::max();

    args.append("d", ref.d);
    args.append("c", ref.c);
    args.append("a", ref.a);
    args.append("b", ref.b);

    args.append("x", ref.x);
    args.append("y", ref.y);
    args.append("z", ref.z);
    args.append("t", ref.t);
    args.append("k", ref.k);

    CHECK(args.size() == sizeof(ref));

    // Check range based-iterator
    for(auto arg : args)
    {
        CHECK(arg.first != nullptr);
        CHECK(arg.second != 0);
    }

    auto begin = args.begin();
    auto end   = args.end();

    // End
    CHECK((begin != end));
    CHECK(end->first == nullptr);
    CHECK(end->second == 0);

    // Pre and post fix operators
    auto postInc = begin++;
    CHECK((postInc != begin));
    CHECK((postInc == args.begin()));
    auto preInc = ++postInc;
    CHECK((preInc == begin));
    CHECK((preInc == postInc));

    // Test logical order
    auto testIt      = args.begin();
    RefT reconstruct = {
        static_cast<void*>(testIt++), // d
        static_cast<void const*>(testIt++), // c
        static_cast<void const*>(testIt++), // a
        static_cast<void const*>(testIt++), // b

        static_cast<int>(testIt++), // x
        static_cast<double>(testIt++), // y
        static_cast<float>(testIt++), // z
        static_cast<char>(testIt++), // t

        static_cast<size_t>(testIt++) // k
    };

    CHECK(ref == reconstruct);
    CHECK((testIt == end));

    // Test throws
    testIt.reset();
    CHECK((testIt == args.begin()));
    char result;
    CHECK_THROWS_AS(result = static_cast<char>(testIt), std::bad_cast);
}

TEMPLATE_TEST_CASE("KernelArguments logging test non pointer numerical values",
                   "[kernel-arguments][logging]",
                   int32_t,
                   int64_t,
                   uint32_t,
                   uint64_t,
                   float,
                   double,
                   Half,
                   FP8,
                   BF8)
{
    using namespace Catch::Matchers;
    TestType value(5.0f);

    KernelArguments args(true);

    args.append("a", value);

    // e.g. [0..3] a: 05 00 00 00 (5)

    auto bytesPart = concatenate("[0..", sizeof(TestType) - 1, "]");

    CHECK_THAT(args.toString(),
               ContainsSubstring(bytesPart) //
                   && ContainsSubstring("a: ") //
                   && ContainsSubstring("5"));
}

TEST_CASE("KernelArguments logging bool", "[kernel-arguments]")
{
    using namespace Catch::Matchers;
    KernelArguments args(true);

    SECTION("false")
    {
        args.append("a", false);

        CHECK_THAT(args.toString(),
                   ContainsSubstring("[0..0]") //
                       && ContainsSubstring("a: ") //
                       && ContainsSubstring("(0)"));
    }

    SECTION("true")
    {
        args.append("a", true);

        CHECK_THAT(args.toString(),
                   ContainsSubstring("0") //
                       && ContainsSubstring("a: ") //
                       && ContainsSubstring("(1)"));
    }
}

TEMPLATE_TEST_CASE("KernelArguments logging test pointer values",
                   "[kernel-arguments][logging]",
                   int32_t,
                   int64_t,
                   uint8_t,
                   uint32_t,
                   uint64_t,
                   float,
                   double,
                   Half,
                   FP8,
                   BF8)
{
    using namespace Catch::Matchers;
    auto value = std::make_shared<TestType>();

    KernelArguments args(true);

    args.append("a", value.get());
    value = nullptr;
    args.append("b", value.get());

    CHECK_THAT(args.toString(),
               ContainsSubstring("[0..7]") //
                   && ContainsSubstring("a: ") //
                   && ContainsSubstring("0x") //
                   && ContainsSubstring("b: ") //
                   && ContainsSubstring("[8..15]") //
                   && (ContainsSubstring("(nil)") //
                       || ContainsSubstring("(0)")));
}
