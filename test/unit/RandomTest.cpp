
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "GenericContextFixture.hpp"
#include "Utilities.hpp"
#include <rocRoller/Utilities/Logging.hpp>
#include <rocRoller/Utilities/Settings.hpp>

class RandomTest : public GenericContextFixture
{
};

using namespace rocRoller;

TEST_F(RandomTest, Seed)
{
    auto seed1 = 12867;
    auto size  = 64;
    auto min   = -10;
    auto max   = 10;

    // Deterministic seed
    auto random1 = RandomGenerator(seed1);
    auto x       = random1.vector<int>(size, min, max);

    EXPECT_EQ(x[0], 0);
    EXPECT_EQ(x[63], 6);

    // Generating again gives something different
    auto x2 = random1.vector<int>(size, min, max);

    EXPECT_NE(x, x2);

    // Re-seeding gives the same vector
    auto random2 = RandomGenerator(seed1);
    auto y       = random2.vector<int>(size, min, max);

    EXPECT_EQ(x, y);

    // Re-seeded generates same sequences
    auto y2 = random2.vector<int>(size, min, max);

    EXPECT_EQ(x2, y2);

    // Test explicit constructor that does not use Settings class
    auto seed2   = 1123;
    auto random3 = RandomGenerator(seed2);
    auto z       = random3.vector<int>(29, -10, 10);

    std::vector<int> Z{8, -7, -3, -3, -2, 3,  1, 6, 6, -2, 3,  -1, 1, 2, -1,
                       6, 2,  -9, -1, 1,  -5, 1, 2, 6, 8,  -2, -3, 7, 5};

    EXPECT_EQ(z, Z);

    // Re-seeding works again
    random3.seed(seed1);
    auto w = random3.vector<int>(size, min, max);

    EXPECT_EQ(x, w);

    // Re-seeded sequence is the same
    auto w2 = random3.vector<int>(size, min, max);

    EXPECT_EQ(x2, w2);

    // Different seeds produce different sequence of vectors
    random1.seed(1729u);
    random2.seed(1730u);
    for(int i = 0; i < 100; ++i)
    {
        auto a = random1.vector<int>(size, min, max);
        auto b = random2.vector<int>(size, min, max);
        EXPECT_NE(a, b);
    }
}

TEST_F(RandomTest, SettingsOverride)
{
    auto size = 8;
    auto min  = -10;
    auto max  = 10;

    int seed         = 8888;
    int settingsSeed = 999999;

    std::vector<int> a, b, c, d;

    {
        auto random_a = RandomGenerator(seed);
        a             = random_a.vector<int>(size, min, max);
    }
    {
        auto random_b = RandomGenerator(settingsSeed);
        b             = random_b.vector<int>(size, min, max);
    }

    Settings::getInstance()->set(Settings::RandomSeed, (int)settingsSeed);

    {
        auto random_c = RandomGenerator(seed); // Arg ignored, settings take priority
        c             = random_c.vector<int>(size, min, max);
    }
    {
        auto random_d = RandomGenerator(1234); // Arg ignored, settings take priority
        d             = random_d.vector<int>(size, min, max);
    }
    EXPECT_NE(a, b);
    EXPECT_NE(a, c);
    EXPECT_NE(a, d);
    EXPECT_EQ(b, c);
    EXPECT_EQ(b, d);

    Settings::getInstance()->reset();
}
