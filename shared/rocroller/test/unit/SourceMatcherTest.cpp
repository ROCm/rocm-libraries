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

#include <gtest/gtest-spi.h>

#include "SimpleFixture.hpp"
#include "SourceMatcher.hpp"

#include <rocRoller/Utilities/Utils.hpp>

using namespace rocRoller;

namespace rocRollerTest
{
    class SourceMatcherTest : public SimpleFixture
    {
    };

    TEST_F(SourceMatcherTest, NormalizedSourceLinesBasic)
    {
        using ::testing::ElementsAre;

        EXPECT_THAT(Generated(NormalizedSourceLines("some code\nmore code", true)),
                    ElementsAre("some code", "more code"));

        EXPECT_THAT(Generated(NormalizedSourceLines(" some   code\nmore   code", true)),
                    ElementsAre("some code", "more code"));

        EXPECT_THAT(Generated(NormalizedSourceLines(" some   code // comment\nmore   code", false)),
                    ElementsAre("some code", "more code"));
    }

    TEST_F(SourceMatcherTest, NormalizedSourceLinesString)
    {
        using ::testing::ElementsAre;

        EXPECT_THAT(Generated(NormalizedSourceLines("some code \"string  constant\"\n"
                                                    "more code",
                                                    true)),
                    ElementsAre("some code \"string  constant\"", "more code"));

        EXPECT_THAT(Generated(NormalizedSourceLines("some code \"unterminated string  constant\n"
                                                    "more code",
                                                    true)),
                    ElementsAre("some code \"unterminated string  constant\nmore code"));

        EXPECT_THAT(Generated(NormalizedSourceLines("some code \"string  \\\"constant  data\\\"\"\n"
                                                    "more code",
                                                    true)),
                    ElementsAre("some code \"string  \\\"constant  data\\\"\"", "more code"));

        EXPECT_THAT(Generated(NormalizedSourceLines("\"nothing but  string constant\"", true)),
                    ElementsAre("\"nothing but  string constant\""));
    }

    TEST_F(SourceMatcherTest, NormalizedSourceLinesComment)
    {
        using ::testing::ElementsAre;
        using ::testing::IsEmpty;

        EXPECT_THAT(Generated(NormalizedSourceLines(" some   code // comment\n"
                                                    "more   code",
                                                    true)),
                    ElementsAre("some code // comment", "more code"));

        EXPECT_THAT(Generated(NormalizedSourceLines(" some   code // comment\n"
                                                    "more   code",
                                                    false)),
                    ElementsAre("some code", "more code"));

        EXPECT_THAT(Generated(NormalizedSourceLines(" some   code // comment\n"
                                                    "more   code // ending comment",
                                                    false)),
                    ElementsAre("some code", "more code"));

        EXPECT_THAT(Generated(NormalizedSourceLines(" some   code // comment\n"
                                                    "more   code // ending comment\n",
                                                    false)),
                    ElementsAre("some code", "more code"));

        EXPECT_THAT(Generated(NormalizedSourceLines(" some   code /* multi\n"
                                                    " line comment */ more   code",
                                                    false)),
                    ElementsAre("some code more code"));

        EXPECT_THAT(Generated(NormalizedSourceLines(" some   code // comment\n"
                                                    "more   code /* unterminated comment",
                                                    false)),
                    ElementsAre("some code", "more code"));

        EXPECT_THAT(Generated(NormalizedSourceLines("// only comments and whitespace\n"
                                                    "   /* another comment */\n  \n",
                                                    false)),
                    IsEmpty());

        EXPECT_THAT(Generated(NormalizedSourceLines("// only comments and whitespace\n"
                                                    "   /* another comment */\n  \n",
                                                    true)),
                    ElementsAre("// only comments and whitespace", "/* another comment */"));
    }

    TEST_F(SourceMatcherTest, NormalizedSourceLinesLonger)
    {
        using ::testing::ElementsAre;

        std::string input
            = ".amdgcn_target \"amdgcn-amd-amdhsa--gfx90a+xnack\"\n"
              "\n"
              ".set   .amdgcn.next_free_vgpr, 0 // Needed if we ever place 2 kernels in one file.\n"
              "   .set .amdgcn.next_free_sgpr, 0\n"
              ".text\n"
              ".globl hello_world\n"
              "   .p2align 8\n"
              ".type hello_world,@function\n"
              "   hello_world:\n\n";

        EXPECT_THAT(
            Generated(NormalizedSourceLines(input, true)),
            ElementsAre(
                ".amdgcn_target \"amdgcn-amd-amdhsa--gfx90a+xnack\"",
                ".set .amdgcn.next_free_vgpr, 0 // Needed if we ever place 2 kernels in one file.",
                ".set .amdgcn.next_free_sgpr, 0",
                ".text",
                ".globl hello_world",
                ".p2align 8",
                ".type hello_world,@function",
                "hello_world:"));

        EXPECT_THAT(Generated(NormalizedSourceLines(input, false)),
                    ElementsAre(".amdgcn_target \"amdgcn-amd-amdhsa--gfx90a+xnack\"",
                                ".set .amdgcn.next_free_vgpr, 0",
                                ".set .amdgcn.next_free_sgpr, 0",
                                ".text",
                                ".globl hello_world",
                                ".p2align 8",
                                ".type hello_world,@function",
                                "hello_world:"));
    }

    TEST_F(SourceMatcherTest, NormalizedSourceLinesYAMLDoc)
    {
#ifdef ROCROLLER_TESTS_USE_YAML_CPP
        using ::testing::ElementsAre;

        std::string input = R"(
.amdgpu_metadata
---
amdhsa.version: [1, 2]
foo:
  baz: 7
  bar: 2
...
.end_amdgpu_metadata

)";
        std::string doc   = R"(---
amdhsa.version:
  - 1
  - 2
foo:
  bar: 2
  baz: 7
...
)";
        EXPECT_THAT(Generated(NormalizedSourceLines(input, false)),
                    ElementsAre(".amdgpu_metadata", doc, ".end_amdgpu_metadata"));
#else
        GTEST_SKIP() << "Skipping NormalizedYAML test (Only implemented for yaml-cpp)";
#endif
    }

    TEST_F(SourceMatcherTest, NormalizedSourceLinesUnterminatedYAMLDoc)
    {
#ifdef ROCROLLER_TESTS_USE_YAML_CPP
        using ::testing::ElementsAre;

        std::string input = R"(
.amdgpu_metadata
---
amdhsa.version:
  - 1
  - 2
)";
        std::string doc   = R"(---
amdhsa.version:
  - 1
  - 2
...
)";
        EXPECT_THAT(Generated(NormalizedSourceLines(input, false)),
                    ElementsAre(".amdgpu_metadata", doc));
#else
        GTEST_SKIP() << "Skipping NormalizedYAML test (Only implemented for yaml-cpp)";
#endif
    }

    TEST_F(SourceMatcherTest, NormalizedYAML)
    {
#ifdef ROCROLLER_TESTS_USE_YAML_CPP
        std::string input1 = "{x: 7, y: 2, a: [4,5,6]}";
        std::string input2 = R"(
x: 7
a: [4,5,6]
y:          2
)";
        std::string input3 = R"(# a comment
x: 7
a: [4,5,6]
y:          2
)";
        std::string input4 = "---\n{x: 7, y: 2, a: [4,5,6]}";

        std::string expected = R"(---
a:
  - 4
  - 5
  - 6
x: 7
y: 2
...
)";
        EXPECT_EQ(expected, NormalizedYAML(input1));
        EXPECT_EQ(expected, NormalizedYAML(input2));
        EXPECT_EQ(expected, NormalizedYAML(input3));
        EXPECT_EQ(expected, NormalizedYAML(input4));

        EXPECT_EQ(NormalizedYAML(input1), NormalizedYAML(NormalizedYAML(input1)));
        EXPECT_EQ(NormalizedYAML(expected), NormalizedYAML(input1));
#else
        GTEST_SKIP() << "Skipping NormalizedYAML test (Only implemented for yaml-cpp)";
#endif
    }

    TEST_F(SourceMatcherTest, MatchesSource)
    {
        std::string programA = "#include <iostream>\n"
                               "int main(int argc, const char * argv[])\n"
                               "{\n"
                               "std::cout << \"Hello,  world!\" << std::endl;\n"
                               "return 0;\n"
                               "}\n";

        std::string programB = "/**\n"
                               " *\n"
                               " * My Awesome program!\n"
                               " *\n"
                               " */\n"
                               "#include <iostream>\n"
                               "int main(int argc, const char * argv[])\n"
                               "{\n"
                               "    std::cout << \"Hello,  world!\" << std::endl;\n"
                               "    return 0; // Success!\n"
                               "}\n";

        std::string programC = "#include <iostream>\n"
                               "int main(int argc, const char * argv[])\n"
                               "{\n"
                               "std::cout << \"Hello, world!\" << std::endl;\n"
                               "return 0;\n"
                               "}\n";

        EXPECT_THAT(programA, MatchesSource(programA));
        EXPECT_THAT(programB, MatchesSource(programB));

        EXPECT_THAT(programB, MatchesSource(programA));
        EXPECT_THAT(programA, MatchesSource(programB));

        EXPECT_NONFATAL_FAILURE(EXPECT_THAT(programC, MatchesSource(programA)), "Hello");
        EXPECT_NONFATAL_FAILURE(EXPECT_THAT(programA, MatchesSource(programC)), "Hello");

        EXPECT_NONFATAL_FAILURE(EXPECT_THAT(programC, MatchesSource(programB)), "Hello");
        EXPECT_NONFATAL_FAILURE(EXPECT_THAT(programB, MatchesSource(programC)), "Hello");
    }

    TEST_F(SourceMatcherTest, MatchesSourceIncludingComments)
    {
        std::string programA = "/**\n"
                               " *\n"
                               " * My Awesome program!\n"
                               " *\n"
                               " */\n"
                               "#include <iostream>\n"
                               "int main(int argc, const char * argv[])\n"
                               "{\n"
                               "std::cout << \"Hello,  world!\" << std::endl;\n"
                               "return 0; // Success!\n"
                               "}\n";

        std::string programB = "/**\n"
                               " *\n"
                               " * My Awesome program!\n"
                               " *\n"
                               " */\n"
                               "#include <iostream>\n"
                               "int main(int argc, const char * argv[])\n"
                               "{\n"
                               "    std::cout << \"Hello,  world!\" << std::endl;\n"
                               "    return 0; // Success!\n"
                               "}\n";

        std::string programC = "#include <iostream>\n"
                               "int main(int argc, const char * argv[])\n"
                               "{\n"
                               "std::cout << \"Hello,  world!\" << std::endl;\n"
                               "return 0;\n"
                               "}\n";

        EXPECT_THAT(programA, MatchesSourceIncludingComments(programA));
        EXPECT_THAT(programB, MatchesSourceIncludingComments(programB));

        EXPECT_THAT(programB, MatchesSourceIncludingComments(programA));
        EXPECT_THAT(programA, MatchesSourceIncludingComments(programB));

        EXPECT_THAT(programC, MatchesSource(programA));
        EXPECT_THAT(programA, MatchesSource(programC));

        EXPECT_NONFATAL_FAILURE(EXPECT_THAT(programC, MatchesSourceIncludingComments(programA)),
                                "Awesome");
        EXPECT_NONFATAL_FAILURE(EXPECT_THAT(programA, MatchesSourceIncludingComments(programC)),
                                "Awesome");

        EXPECT_NONFATAL_FAILURE(EXPECT_THAT(programC, MatchesSourceIncludingComments(programB)),
                                "Awesome");
        EXPECT_NONFATAL_FAILURE(EXPECT_THAT(programB, MatchesSourceIncludingComments(programC)),
                                "Awesome");
    }

}
