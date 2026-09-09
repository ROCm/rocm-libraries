// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

// Value is the language's only runtime type, and every operator's semantics
// route through it. The suites in TestJsonExpression.cpp reach it through
// compiled rules, which cannot address its boundaries directly: the 2^53 fold
// in number(), the exact int/double comparison ladder, and every producer of
// Ordering::UNORDERED. Those are tested here, against Value itself.

#include <gtest/gtest.h>

#include <array>
#include <clocale>
#include <cmath>
#include <cstdint>
#include <limits>
#include <sstream>
#include <string>

#include <hipdnn_plugin_sdk/ingestor/jsonexpr/Value.hpp>

namespace jexpr = hipdnn_plugin_sdk::ingestor::jsonexpr;

using V = jexpr::Value;
using Ordering = V::Ordering;

namespace
{
constexpr double TWO_POW_53 = 9007199254740992.0;
constexpr auto MAX_INT64 = std::numeric_limits<std::int64_t>::max();

/// Read a string through the same path an operator does: toNumber() on a
/// string value is the only caller of the numeric-string parser.
double asNumber(const std::string& s)
{
    return V(s).toNumber();
}
} // namespace

// ---------------------------------------------------------------------------
// Kinds and construction.
// ---------------------------------------------------------------------------
TEST(TestJsonValue, KindPredicatesPartitionTheAlternatives)
{
    EXPECT_TRUE(V().isNull());
    EXPECT_TRUE(V(nullptr).isNull());
    EXPECT_TRUE(V(true).isBool());
    EXPECT_TRUE(V(std::int64_t{7}).isInt());
    EXPECT_TRUE(V(1.5).isDouble());
    EXPECT_TRUE(V("s").isString());
    EXPECT_TRUE(V(std::string("s")).isString());
    EXPECT_TRUE(V(V::Array{}).isArray());

    // isNumber covers both numeric alternatives and nothing else.
    EXPECT_TRUE(V(std::int64_t{7}).isNumber());
    EXPECT_TRUE(V(1.5).isNumber());
    EXPECT_FALSE(V("7").isNumber());
    EXPECT_FALSE(V(true).isNumber());
    EXPECT_FALSE(V().isNumber());

    // An int is stored as an int64, not as a double, so a plain literal in a
    // rule compares exactly against a data source's integer.
    EXPECT_TRUE(V(7).isInt());
    EXPECT_EQ(V(7).asInt(), 7);
}

TEST(TestJsonValue, NumberFoldsAnExactlyIntegralDoubleToAnInteger)
{
    // Integral and representable: folds, so arithmetic over integers yields
    // integers and stays exactly comparable afterwards.
    EXPECT_TRUE(V::number(4.0).isInt());
    EXPECT_EQ(V::number(4.0).asInt(), 4);
    EXPECT_TRUE(V::number(-4.0).isInt());

    // Fractional: stays a double.
    EXPECT_TRUE(V::number(1.5).isDouble());

    // The fold boundary. 2^53 is the largest magnitude at which every integral
    // double is still exactly one int64, so it folds and the next power does
    // not.
    EXPECT_TRUE(V::number(TWO_POW_53).isInt());
    EXPECT_EQ(V::number(TWO_POW_53).asInt(), std::int64_t{1} << 53);
    EXPECT_TRUE(V::number(-TWO_POW_53).isInt());
    EXPECT_TRUE(V::number(TWO_POW_53 * 2.0).isDouble());

    // Non-finite input is never folded; it stays a double so the finiteness
    // gate in Operators.hpp can still see it.
    EXPECT_TRUE(V::number(std::nan("")).isDouble());
    EXPECT_TRUE(V::number(std::numeric_limits<double>::infinity()).isDouble());
}

// ---------------------------------------------------------------------------
// Truthiness and numeric coercion.
// ---------------------------------------------------------------------------
TEST(TestJsonValue, TruthinessFollowsJavaScript)
{
    EXPECT_FALSE(V().truthy());
    EXPECT_FALSE(V(false).truthy());
    EXPECT_FALSE(V(std::int64_t{0}).truthy());
    EXPECT_FALSE(V(0.0).truthy());
    EXPECT_FALSE(V("").truthy());
    EXPECT_FALSE(V(V::Array{}).truthy());

    EXPECT_TRUE(V(true).truthy());
    EXPECT_TRUE(V(std::int64_t{-1}).truthy());
    EXPECT_TRUE(V(0.5).truthy());
    EXPECT_TRUE(V("0").truthy()); // a non-empty string, however it reads
    EXPECT_TRUE(V(V::Array{V(0)}).truthy());
}

TEST(TestJsonValue, ToNumberCoercesEachAlternative)
{
    EXPECT_EQ(V().toNumber(), 0.0);
    EXPECT_EQ(V(true).toNumber(), 1.0);
    EXPECT_EQ(V(false).toNumber(), 0.0);
    EXPECT_EQ(V(std::int64_t{7}).toNumber(), 7.0);
    EXPECT_EQ(V(1.5).toNumber(), 1.5);

    // An empty array is 0 and a one-element array is its element, as Number()
    // has it. More than one element has no numeric reading, so it is NaN,
    // which the finiteness gate turns into a decline.
    EXPECT_EQ(V(V::Array{}).toNumber(), 0.0);
    EXPECT_EQ(V(V::Array{V(3)}).toNumber(), 3.0);
    EXPECT_TRUE(std::isnan(V(V::Array{V(1), V(2)}).toNumber()));
}

// ---------------------------------------------------------------------------
// Numeric strings. from_chars, not strtod: see Value::stringToNumber.
// ---------------------------------------------------------------------------
TEST(TestJsonValue, NumericStringsAcceptTheOrdinarySpellings)
{
    EXPECT_EQ(asNumber("5"), 5.0);
    EXPECT_EQ(asNumber("-5"), -5.0);
    EXPECT_EQ(asNumber("+5"), 5.0); // restored by hand; from_chars rejects it
    EXPECT_EQ(asNumber("1.5"), 1.5);
    EXPECT_EQ(asNumber("1e3"), 1000.0);
    EXPECT_EQ(asNumber("  5  "), 5.0); // surrounding whitespace is trimmed
    EXPECT_EQ(asNumber(""), 0.0); // Number("") == 0
    EXPECT_EQ(asNumber("   "), 0.0); // trims to empty
}

TEST(TestJsonValue, NumericStringsRejectEverythingElse)
{
    EXPECT_TRUE(std::isnan(asNumber("abc")));
    EXPECT_TRUE(std::isnan(asNumber("5abc"))); // trailing garbage
    EXPECT_TRUE(std::isnan(asNumber("0x10"))); // hex is a typo, not 16
    EXPECT_TRUE(std::isnan(asNumber("+"))); // a bare sign is not a number
    EXPECT_TRUE(std::isnan(asNumber("-")));
    EXPECT_TRUE(std::isnan(asNumber("+-5"))); // only one leading sign is accepted
    EXPECT_TRUE(std::isnan(asNumber("++5")));
    EXPECT_TRUE(std::isnan(asNumber("-+5")));
    EXPECT_TRUE(std::isnan(asNumber("--5")));
    EXPECT_TRUE(std::isnan(asNumber("1 5"))); // interior whitespace
}

TEST(TestJsonValue, NumericStringUnderflowDeclinesInsteadOfReadingAsZero)
{
    // The bug this pins: strtod reports underflow only through errno, which
    // went unchecked, so "1e-999" read as a clean 0. A criterion comparing it
    // against 0 then answered a confident true about a value the language
    // never represented.
    EXPECT_TRUE(std::isnan(asNumber("1e-999")));
    EXPECT_TRUE(std::isnan(asNumber("-1e-999")));
    EXPECT_NE(V("1e-999"), V(0));
    EXPECT_EQ(V::compare(V("1e-999"), V(0)), Ordering::UNORDERED);

    // A denormal is representable, so it is a value and not an error. errno
    // cannot tell these two apart; from_chars can.
    EXPECT_EQ(asNumber("5e-324"), std::numeric_limits<double>::denorm_min());
    EXPECT_GT(asNumber("5e-324"), 0.0);

    // Overflow was already caught downstream, because infinity is not finite.
    EXPECT_FALSE(std::isfinite(asNumber("1e999")));
}

TEST(TestJsonValue, NumericStringParsingIsLocaleIndependent)
{
    // strtod reads "1.5" as 1 under a comma-decimal LC_NUMERIC. A library
    // evaluating rules read off disk does not control the host process's
    // locale, so the parser must not consult it.
    const char* previous = std::setlocale(LC_NUMERIC, nullptr);
    const std::string saved = (previous != nullptr) ? previous : "C";
    if(std::setlocale(LC_NUMERIC, "de_DE.UTF-8") == nullptr)
    {
        GTEST_SKIP() << "no comma-decimal locale on this image";
    }
    EXPECT_EQ(asNumber("1.5"), 1.5);
    EXPECT_TRUE(std::isnan(asNumber("1,5"))); // and the local spelling is not one
    std::setlocale(LC_NUMERIC, saved.c_str());
}

// ---------------------------------------------------------------------------
// Equality.
// ---------------------------------------------------------------------------
TEST(TestJsonValue, EqualityComparesTwoIntegersExactly)
{
    EXPECT_EQ(V(MAX_INT64), V(MAX_INT64));

    // Adjacent int64 above 2^53. Routed through double these conflate, and
    // this language gates dispatch on sizes, strides and byte offsets, so that
    // would be a wrong decision rather than a rounding error.
    const std::int64_t big = std::int64_t{1} << 60;
    EXPECT_NE(V(big), V(big + 1));
    EXPECT_EQ(static_cast<double>(big), static_cast<double>(big + 1)); // the trap avoided
}

TEST(TestJsonValue, EqualityAcrossIntAndDoubleHoldsOnlyWhenExact)
{
    EXPECT_EQ(V(4), V(4.0));
    EXPECT_EQ(V(4.0), V(4));
    EXPECT_NE(V(4), V(4.5));

    // An integer beyond the double's reach is not equal to the double it
    // would round to.
    const std::int64_t big = (std::int64_t{1} << 60) + 1;
    EXPECT_NE(V(big), V(static_cast<double>(big)));

    // A non-finite double equals no integer.
    EXPECT_NE(V(0), V(std::nan("")));
    EXPECT_NE(V(0), V(std::numeric_limits<double>::infinity()));
}

TEST(TestJsonValue, EqualityIsStrictAcrossKinds)
{
    EXPECT_NE(V("1"), V(1));
    EXPECT_NE(V(1), V("1"));
    EXPECT_NE(V(1), V(true));
    EXPECT_NE(V(0), V(V::Array{}));
    EXPECT_NE(V(""), V());

    // Two nulls are equal *here*, because operator== is plain variant
    // equality. The decline lives one layer up: OpNode::eval gates on
    // containsUnresolved, so `==` never sees an unresolved operand and a rule
    // comparing two absent paths yields null rather than true. The rule-level
    // behaviour is pinned in TestJsonExpression.NullPropagation.
    EXPECT_EQ(V(), V());
    EXPECT_TRUE(V().containsUnresolved());

    // Arrays compare element-wise, by these same rules.
    EXPECT_EQ(V(V::Array{V(1), V("a")}), V(V::Array{V(1), V("a")}));
    EXPECT_NE(V(V::Array{V(1)}), V(V::Array{V(1), V(2)}));
    EXPECT_NE(V(V::Array{V(1)}), V(V::Array{V("1")}));
}

// ---------------------------------------------------------------------------
// Ordering.
// ---------------------------------------------------------------------------
TEST(TestJsonValue, CompareOrdersLikeKinds)
{
    EXPECT_EQ(V::compare(V(1), V(2)), Ordering::LESS);
    EXPECT_EQ(V::compare(V(2), V(1)), Ordering::GREATER);
    EXPECT_EQ(V::compare(V(2), V(2)), Ordering::EQUAL);

    EXPECT_EQ(V::compare(V("abc"), V("abd")), Ordering::LESS);
    EXPECT_EQ(V::compare(V("abd"), V("abc")), Ordering::GREATER);
    EXPECT_EQ(V::compare(V("abc"), V("abc")), Ordering::EQUAL);

    EXPECT_EQ(V::compare(V(1), V(1.5)), Ordering::LESS);
    EXPECT_EQ(V::compare(V(1.5), V(1)), Ordering::GREATER);
    EXPECT_EQ(V::compare(V(2), V(2.0)), Ordering::EQUAL);

    // Bools and arrays reach the coercing tail, as they always have.
    EXPECT_EQ(V::compare(V(false), V(true)), Ordering::LESS);
    EXPECT_EQ(V::compare(V(V::Array{V(1)}), V(2)), Ordering::LESS);
}

TEST(TestJsonValue, CompareOrdersIntegersWithoutRoundingThroughDouble)
{
    const std::int64_t big = std::int64_t{1} << 60;
    EXPECT_EQ(V::compare(V(big), V(big + 1)), Ordering::LESS);
    EXPECT_EQ(V::compare(V(big + 1), V(big)), Ordering::GREATER);

    // int64 against double, on both sides of the double's exact range.
    EXPECT_EQ(V::compare(V(MAX_INT64), V(1e300)), Ordering::LESS);
    EXPECT_EQ(V::compare(V(1e300), V(MAX_INT64)), Ordering::GREATER);
    EXPECT_EQ(V::compare(V(MAX_INT64), V(-1e300)), Ordering::GREATER);
    EXPECT_EQ(V::compare(V(4), V(4.5)), Ordering::LESS);
    EXPECT_EQ(V::compare(V(4.5), V(4)), Ordering::GREATER);
}

TEST(TestJsonValue, CompareDeclinesOnAKindMismatchSoOrderingAgreesWithEquality)
{
    // The asymmetry this pins: ordering used to coerce a string operand, so
    // "1" == 1 was false while "1" <= 1 was true. One expression reporting a
    // pair as unequal *and* as ordered is a contradiction, and the coercing
    // answer is the widening one.
    EXPECT_NE(V("1"), V(1));
    EXPECT_EQ(V::compare(V("1"), V(1)), Ordering::UNORDERED);
    EXPECT_EQ(V::compare(V(1), V("1")), Ordering::UNORDERED);

    // It holds however the number is spelled, and for a string that has no
    // numeric reading at all.
    EXPECT_EQ(V::compare(V("1.0"), V(1.0)), Ordering::UNORDERED);
    EXPECT_EQ(V::compare(V("amd"), V(8)), Ordering::UNORDERED);

    // A string against a non-number is a kind mismatch too.
    EXPECT_EQ(V::compare(V("1"), V(true)), Ordering::UNORDERED);
    EXPECT_EQ(V::compare(V("1"), V()), Ordering::UNORDERED);
    EXPECT_EQ(V::compare(V("1"), V(V::Array{V(1)})), Ordering::UNORDERED);

    // Arithmetic still coerces a numeric string: only the predicates that gate
    // criteria are strict.
    EXPECT_EQ(V("2").toNumber() + V("3").toNumber(), 5.0);
}

TEST(TestJsonValue, CompareDeclinesOnANonFiniteOperand)
{
    const V nan(std::nan(""));
    const V inf(std::numeric_limits<double>::infinity());

    EXPECT_EQ(V::compare(nan, V(1.0)), Ordering::UNORDERED);
    EXPECT_EQ(V::compare(V(1.0), nan), Ordering::UNORDERED);
    EXPECT_EQ(V::compare(V(1), nan), Ordering::UNORDERED);
    EXPECT_EQ(V::compare(nan, V(1)), Ordering::UNORDERED);
    EXPECT_EQ(V::compare(inf, inf), Ordering::UNORDERED);

    // A multi-element array coerces to NaN, so it is unorderable as well.
    EXPECT_EQ(V::compare(V(V::Array{V(1), V(2)}), V(1.0)), Ordering::UNORDERED);
}

TEST(TestJsonValue, CompareIsAntisymmetric)
{
    // Every ladder branch must reverse cleanly, including the two that route
    // through reverse() rather than comparing directly.
    const std::array samples = {V(1),
                                V(2),
                                V(1.5),
                                V(MAX_INT64),
                                V(1e300),
                                V("a"),
                                V("b"),
                                V(true),
                                V(),
                                V(std::nan("")),
                                V(V::Array{V(1)})};
    for(const V& a : samples)
    {
        for(const V& b : samples)
        {
            const Ordering forward = V::compare(a, b);
            const Ordering backward = V::compare(b, a);
            switch(forward)
            {
            case Ordering::LESS:
                EXPECT_EQ(backward, Ordering::GREATER) << a << " vs " << b;
                break;
            case Ordering::GREATER:
                EXPECT_EQ(backward, Ordering::LESS) << a << " vs " << b;
                break;
            case Ordering::EQUAL:
            case Ordering::UNORDERED:
            default:
                EXPECT_EQ(backward, forward) << a << " vs " << b;
                break;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Unresolved propagation.
// ---------------------------------------------------------------------------
TEST(TestJsonValue, ContainsUnresolvedFindsANullAtAnyDepth)
{
    EXPECT_TRUE(V().containsUnresolved());
    EXPECT_FALSE(V(0).containsUnresolved());
    EXPECT_FALSE(V("").containsUnresolved());
    EXPECT_FALSE(V(V::Array{}).containsUnresolved());

    EXPECT_TRUE(V(V::Array{V(1), V()}).containsUnresolved());
    EXPECT_FALSE(V(V::Array{V(1), V(2)}).containsUnresolved());

    // Nested, because a stride_order can arrive as an array of arrays and a
    // hole anywhere in it means the value was only partly read.
    EXPECT_TRUE(V(V::Array{V(V::Array{V(1), V()})}).containsUnresolved());
    EXPECT_FALSE(V(V::Array{V(V::Array{V(1), V(2)})}).containsUnresolved());
}

// ---------------------------------------------------------------------------
// Rendering. GoogleTest failure output depends on these.
// ---------------------------------------------------------------------------
TEST(TestJsonValue, DumpRendersEachAlternative)
{
    EXPECT_EQ(V().dump(), "null");
    EXPECT_EQ(V(true).dump(), "true");
    EXPECT_EQ(V(false).dump(), "false");
    EXPECT_EQ(V(std::int64_t{-7}).dump(), "-7");
    EXPECT_EQ(V("s").dump(), "\"s\"");
    EXPECT_EQ(V(V::Array{V(1), V("s"), V()}).dump(), "[1,\"s\",null]");
    EXPECT_EQ(V(V::Array{}).dump(), "[]");

    std::ostringstream os;
    os << V(V::Array{V(1), V()});
    EXPECT_EQ(os.str(), V(V::Array{V(1), V()}).dump());
}

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
