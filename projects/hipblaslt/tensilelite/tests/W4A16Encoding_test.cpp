// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>
#include <map>
#include <string>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>
#ifdef TENSILE_MSGPACK
#include <Tensile/msgpack/MessagePack.hpp>
#endif

using namespace TensileLite;
using Encoding = ContractionProblemGemm::Int4Encoding;

TEST(W4A16Encoding, PredicateMatchesEncoding)
{
    ContractionProblemGemm problem;
    Predicates::Contraction::Int4EncodingA signedPredicate(Encoding::Signed);
    Predicates::Contraction::Int4EncodingA unsignedPredicate(Encoding::UnsignedBias8);
    EXPECT_EQ(ContractionSolution::ProblemType{}.int4EncodingA, Encoding::Signed);
    EXPECT_EQ(Predicates::Contraction::Int4EncodingA{}.value, Encoding::Signed);
    for(auto encoding : {Encoding::Signed, Encoding::UnsignedBias8})
    {
        problem.setInt4EncodingA(encoding);
        EXPECT_EQ(signedPredicate(problem), encoding == Encoding::Signed);
        EXPECT_EQ(unsignedPredicate(problem), encoding == Encoding::UnsignedBias8);
    }
}

#ifdef TENSILE_MSGPACK
TEST(W4A16Encoding, ExistingSerializedNamesLoad)
{
    for(auto const& name : {"Signed", "UnsignedBias8"})
    {
        const auto expected = std::string(name) == "Signed" ? Encoding::Signed
                                                            : Encoding::UnsignedBias8;
        msgpack::zone zone;
        Encoding value = Encoding::Signed;
        Serialization::MessagePackInput scalar(msgpack::object(std::string(name), zone));
        scalar.input(value);
        EXPECT_TRUE(scalar.error.empty());
        EXPECT_EQ(value, expected);

        std::map<std::string, std::string> fields{{"type", "Int4EncodingA"}, {"value", name}};
        Serialization::MessagePackInput reader(msgpack::object(fields, zone));
        Predicates::Contraction::Int4EncodingA predicate;
        reader.input(predicate);
        ASSERT_TRUE(reader.error.empty());
        EXPECT_EQ(predicate.value, expected);
    }
}

TEST(W4A16Encoding, UnknownSerializedNameIsRejected)
{
    msgpack::zone zone;
    Encoding value = Encoding::Signed;
    Serialization::MessagePackInput reader(msgpack::object(std::string("Unknown"), zone));
    reader.input(value);
    EXPECT_FALSE(reader.error.empty());
}
#endif
