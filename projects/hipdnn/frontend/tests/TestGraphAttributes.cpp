// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/attributes/GraphAttributes.hpp>

TEST(TestGraphAttributes, CreateGraphAttributes)
{
    hipdnn_frontend::graph::GraphAttributes graphAttributes;

    // Set all properties
    graphAttributes.set_name("TestGraph")
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::HALF)
        .set_io_data_type(hipdnn_frontend::DataType::BFLOAT16);

    // Verify all properties
    EXPECT_EQ(graphAttributes.get_name(), "TestGraph");
    EXPECT_EQ(graphAttributes.get_compute_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(graphAttributes.get_intermediate_data_type(), hipdnn_frontend::DataType::HALF);
    EXPECT_EQ(graphAttributes.get_io_data_type(), hipdnn_frontend::DataType::BFLOAT16);
}

TEST(TestGraphAttributes, DefaultValues)
{
    hipdnn_frontend::graph::GraphAttributes graphAttributes;

    // Check default values
    EXPECT_EQ(graphAttributes.get_name(), "");
    EXPECT_EQ(graphAttributes.get_compute_data_type(), hipdnn_frontend::DataType::NOT_SET);
    EXPECT_EQ(graphAttributes.get_intermediate_data_type(), hipdnn_frontend::DataType::NOT_SET);
    EXPECT_EQ(graphAttributes.get_io_data_type(), hipdnn_frontend::DataType::NOT_SET);
}

TEST(TestGraphAttributes, SetName)
{
    hipdnn_frontend::graph::GraphAttributes graphAttributes;

    graphAttributes.set_name("MyCustomGraph");
    EXPECT_EQ(graphAttributes.get_name(), "MyCustomGraph");

    // Test empty string
    graphAttributes.set_name("");
    EXPECT_EQ(graphAttributes.get_name(), "");

    // Test long name
    std::string longName = "VeryLongGraphNameThatShouldStillWorkProperly";
    graphAttributes.set_name(longName);
    EXPECT_EQ(graphAttributes.get_name(), longName);
}

TEST(TestGraphAttributes, SetComputeDataType)
{
    hipdnn_frontend::graph::GraphAttributes graphAttributes;

    graphAttributes.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(graphAttributes.get_compute_data_type(), hipdnn_frontend::DataType::FLOAT);
}

TEST(TestGraphAttributes, SetIntermediateDataType)
{
    hipdnn_frontend::graph::GraphAttributes graphAttributes;

    graphAttributes.set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(graphAttributes.get_intermediate_data_type(), hipdnn_frontend::DataType::FLOAT);
}

TEST(TestGraphAttributes, SetIoDataType)
{
    hipdnn_frontend::graph::GraphAttributes graphAttributes;

    graphAttributes.set_io_data_type(hipdnn_frontend::DataType::HALF);
    EXPECT_EQ(graphAttributes.get_io_data_type(), hipdnn_frontend::DataType::HALF);
}

TEST(TestGraphAttributes, MethodChaining)
{
    hipdnn_frontend::graph::GraphAttributes graphAttributes;

    // Test that all setters return reference to self for chaining
    auto& ref1 = graphAttributes.set_name("ChainedGraph");
    auto& ref2 = ref1.set_compute_data_type(hipdnn_frontend::DataType::FLOAT);
    auto& ref3 = ref2.set_intermediate_data_type(hipdnn_frontend::DataType::HALF);
    auto& ref4 = ref3.set_io_data_type(hipdnn_frontend::DataType::BFLOAT16);

    // All references should point to the same object
    EXPECT_EQ(&graphAttributes, &ref1);
    EXPECT_EQ(&graphAttributes, &ref2);
    EXPECT_EQ(&graphAttributes, &ref3);
    EXPECT_EQ(&graphAttributes, &ref4);

    // Verify all values were set correctly
    EXPECT_EQ(graphAttributes.get_name(), "ChainedGraph");
    EXPECT_EQ(graphAttributes.get_compute_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(graphAttributes.get_intermediate_data_type(), hipdnn_frontend::DataType::HALF);
    EXPECT_EQ(graphAttributes.get_io_data_type(), hipdnn_frontend::DataType::BFLOAT16);
}

TEST(TestGraphAttributes, FillMissingPropertiesAllMissing)
{
    hipdnn_frontend::graph::GraphAttributes target;
    hipdnn_frontend::graph::GraphAttributes source;

    // Set all properties in source
    source.set_name("SourceGraph")
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::HALF)
        .set_io_data_type(hipdnn_frontend::DataType::BFLOAT16);

    target.fill_missing_properties(source);

    // All properties should be copied from source
    EXPECT_EQ(target.get_name(), "SourceGraph");
    EXPECT_EQ(target.get_compute_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(target.get_intermediate_data_type(), hipdnn_frontend::DataType::HALF);
    EXPECT_EQ(target.get_io_data_type(), hipdnn_frontend::DataType::BFLOAT16);
}

TEST(TestGraphAttributes, FillMissingPropertiesPartialMissing)
{
    hipdnn_frontend::graph::GraphAttributes target;
    hipdnn_frontend::graph::GraphAttributes source;

    // Set some properties in target
    target.set_name("TargetGraph").set_compute_data_type(hipdnn_frontend::DataType::DOUBLE);

    // Set all properties in source
    source.set_name("SourceGraph")
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::HALF)
        .set_io_data_type(hipdnn_frontend::DataType::BFLOAT16);

    // Fill missing properties
    target.fill_missing_properties(source);

    // Existing properties should not be overwritten
    EXPECT_EQ(target.get_name(), "TargetGraph");
    EXPECT_EQ(target.get_compute_data_type(), hipdnn_frontend::DataType::DOUBLE);

    // Missing properties should be filled from source
    EXPECT_EQ(target.get_intermediate_data_type(), hipdnn_frontend::DataType::HALF);
    EXPECT_EQ(target.get_io_data_type(), hipdnn_frontend::DataType::BFLOAT16);
}

TEST(TestGraphAttributes, FillMissingPropertiesNoneMissing)
{
    hipdnn_frontend::graph::GraphAttributes target;
    hipdnn_frontend::graph::GraphAttributes source;

    // Set all properties in target
    target.set_name("TargetGraph")
        .set_compute_data_type(hipdnn_frontend::DataType::DOUBLE)
        .set_intermediate_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_io_data_type(hipdnn_frontend::DataType::INT32);

    // Set all properties in source with different values
    source.set_name("SourceGraph")
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::HALF)
        .set_io_data_type(hipdnn_frontend::DataType::BFLOAT16);

    // Fill missing properties (none are missing)
    target.fill_missing_properties(source);

    // No properties should be changed
    EXPECT_EQ(target.get_name(), "TargetGraph");
    EXPECT_EQ(target.get_compute_data_type(), hipdnn_frontend::DataType::DOUBLE);
    EXPECT_EQ(target.get_intermediate_data_type(), hipdnn_frontend::DataType::FLOAT);
    EXPECT_EQ(target.get_io_data_type(), hipdnn_frontend::DataType::INT32);
}

TEST(TestGraphAttributes, FillMissingPropertiesEmptyName)
{
    hipdnn_frontend::graph::GraphAttributes target;
    hipdnn_frontend::graph::GraphAttributes source;

    // Set empty name in target and non-empty in source
    target.set_name("").set_compute_data_type(hipdnn_frontend::DataType::FLOAT);

    source.set_name("SourceGraph").set_compute_data_type(hipdnn_frontend::DataType::DOUBLE);

    // Fill missing properties
    target.fill_missing_properties(source);

    // Empty name should be filled from source
    EXPECT_EQ(target.get_name(), "SourceGraph");
    // Existing compute type should not change
    EXPECT_EQ(target.get_compute_data_type(), hipdnn_frontend::DataType::FLOAT);
}

TEST(TestGraphAttributes, FillMissingPropertiesChaining)
{
    hipdnn_frontend::graph::GraphAttributes target;
    hipdnn_frontend::graph::GraphAttributes source;

    source.set_name("SourceGraph")
        .set_compute_data_type(hipdnn_frontend::DataType::FLOAT)
        .set_intermediate_data_type(hipdnn_frontend::DataType::HALF)
        .set_io_data_type(hipdnn_frontend::DataType::BFLOAT16);

    // Test that fill_missing_properties returns reference for chaining
    auto& ref = target.fill_missing_properties(source);
    EXPECT_EQ(&target, &ref);

    // Chain additional operations
    ref.set_name("ModifiedAfterFill");

    EXPECT_EQ(target.get_name(), "ModifiedAfterFill");
    EXPECT_EQ(target.get_compute_data_type(), hipdnn_frontend::DataType::FLOAT);
}
