#include <gtest/gtest.h>

#include <hipdnn_sdk/utilities/Visitor.hpp>
#include <string>
#include <variant>

using namespace hipdnn_sdk::utilities;

TEST(TestVisitor, Correctness)
{
    using VariantType = std::variant<float, int, char, std::string>;
    auto visitor = hipdnn_sdk::utilities::Visitor{
        [](float) { return 0; }, [](int) { return 1; }, [](const auto&) { return 2; }};

    EXPECT_EQ(std::visit(visitor, VariantType{0.f}), 0);
    EXPECT_EQ(std::visit(visitor, VariantType{1}), 1);
    EXPECT_EQ(std::visit(visitor, VariantType{'c'}), 2);
    EXPECT_EQ(std::visit(visitor, VariantType{std::string{"abc"}}), 2);
}
