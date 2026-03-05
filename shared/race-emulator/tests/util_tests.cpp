#include "race-emulator/Util.h"
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <string>

using namespace raceemulator;

// PEEPHOLE BUG 3: signed integer overflow (caught by UBSAN only)
TEST(UtilTests, PeepholeBugUbsan) {
  int x = std::numeric_limits<int>::max();
  x += 1; // undefined behavior: signed overflow
  EXPECT_NE(x, 0);
}

// PEEPHOLE BUG 2: heap-buffer-overflow (caught by ASAN only)
TEST(UtilTests, PeepholeBugAsan) {
  int size = 4;
  int *p = new int[size];
  for (int i = 0; i <= size; i++) // off-by-one: writes p[4]
    p[i] = i;
  EXPECT_EQ(p[0], 0);
  delete[] p;
}

TEST(UtilTests, SymbolArithmeticTest0) {
  const std::map<std::string, uint32_t> table = {{"present", 4}};
  const std::string line =
      "v_xx v[foo:bar], s[3 +present  + 2*6 + 0 * 0], v[buzz]";
  std::string reduced = getSymbolReducedLine(line, table);
  //   3 + present + 2*6
  // = 3 + 4       + 12
  // = 19
  // Expect the substring s19 to appear in the reduced line.
  EXPECT_NE(reduced.find("s19"), std::string::npos);
}
TEST(ParserUtils, ParsePackedModifiersStrictFormat) {
  // 1. Valid Standard Case
  std::string line1 = "v_pk_mul_f32 v0, v1, v2 op_sel:[1,0]";
  auto mods1 = parsePackedModifiers(line1);
  ASSERT_EQ(mods1.size(), 1);
  EXPECT_EQ(mods1["op_sel"][0], 1);

  // 2. Invalid Space after Colon (Strict Check)
  // "neg_lo: [1,0]" -> '[' is not at colon+1. -> IGNORED
  std::string line2 = "v_add v0, v1 neg_lo: [1,0]";
  auto mods2 = parsePackedModifiers(line2);
  EXPECT_TRUE(mods2.empty());

  // 3. Register Range (Ignored)
  // "v[0:1]" -> Colon at 3. Char at 4 is '1', not '['. -> IGNORED
  std::string line3 = "v_mov_b64 v[0:1], s[2:3]";
  auto mods3 = parsePackedModifiers(line3);
  EXPECT_TRUE(mods3.empty());

  // 4. Scalar Offset (Ignored)
  // "offset:16" -> Colon followed by '1'. -> IGNORED
  std::string line4 = "buffer_load v0, offset:16";
  auto mods4 = parsePackedModifiers(line4);
  EXPECT_TRUE(mods4.empty());

  // 5. Hex Support
  std::string line5 = "test key:[0xA, 2]";
  auto mods5 = parsePackedModifiers(line5);
  EXPECT_EQ(mods5["key"][0], 10);
  EXPECT_EQ(mods5["key"][1], 2);
}
