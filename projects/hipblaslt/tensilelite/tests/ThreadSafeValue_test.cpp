// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <thread>
#include <vector>

#include <Tensile/Utils.hpp>

using TensileLite::ThreadSafeValue;

namespace
{
    // Counts copies, so a test can see whether an assignment stored a value.
    struct CopyCounted
    {
        static int copies;

        CopyCounted() = default;

        explicit CopyCounted(int v)
            : value(v)
        {
        }

        CopyCounted(CopyCounted const& other)
            : value(other.value)
        {
            ++copies;
        }

        CopyCounted& operator=(CopyCounted const&) = default;

        bool operator==(CopyCounted const& other) const
        {
            return value == other.value;
        }

        int value = 0;
    };

    int CopyCounted::copies = 0;
}

TEST(ThreadSafeValueTest, UnassignedValueLoadsDefault)
{
    ThreadSafeValue<std::string> unassigned;
    EXPECT_EQ(unassigned.load(), "");

    ThreadSafeValue<std::string> copied(unassigned);
    EXPECT_EQ(copied.load(), "");

    ThreadSafeValue<std::string> assigned(std::string("name"));
    assigned = unassigned;
    EXPECT_EQ(assigned.load(), "");
}

TEST(ThreadSafeValueTest, AssigningTheCurrentValueStoresNothing)
{
    ThreadSafeValue<CopyCounted> value;
    CopyCounted const            one(1);

    CopyCounted::copies = 0;
    value               = one;
    EXPECT_EQ(CopyCounted::copies, 1);

    value = one;
    EXPECT_EQ(CopyCounted::copies, 1) << "the value already held is not stored again";

    value = CopyCounted(2);
    EXPECT_EQ(CopyCounted::copies, 2);
    EXPECT_EQ(value.load().value, 2);
}

TEST(ThreadSafeValueTest, AlternatingValuesKeepOneCopyEach)
{
    ThreadSafeValue<CopyCounted> value;
    CopyCounted const            one(1);
    CopyCounted const            two(2);

    CopyCounted::copies = 0;
    for(int i = 0; i < 100; ++i)
        value = i % 2 ? two : one;
    EXPECT_EQ(CopyCounted::copies, 2) << "a value held before is republished, not copied";

    CopyCounted::copies = 0;
    EXPECT_EQ(value.load().value, 2);
    value = one;
    EXPECT_EQ(value.load().value, 1);
    EXPECT_EQ(CopyCounted::copies, 2) << "only the two loads copy";
}

TEST(ThreadSafeValueTest, ReadersSeeOnlyWholeValues)
{
    // Longer than any small-string buffer, so each value is a heap copy.
    std::string const first(64, 'a');
    std::string const second(64, 'b');

    ThreadSafeValue<std::string> value(first);
    std::atomic<bool>            done{false};
    std::atomic<bool>            torn{false};

    std::vector<std::thread> readers;
    for(int t = 0; t < 4; ++t)
        readers.emplace_back([&] {
            while(!done.load())
            {
                std::string const seen = value.load();
                if(seen != first && seen != second)
                    torn = true;
            }
        });

    for(int i = 0; i < 2000; ++i)
        value = i % 2 ? first : second;

    done = true;
    for(auto& r : readers)
        r.join();

    EXPECT_FALSE(torn.load());
    EXPECT_EQ(value.load(), first);
}
