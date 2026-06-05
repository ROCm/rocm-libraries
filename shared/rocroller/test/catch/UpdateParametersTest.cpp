// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <rocRoller/Expression.hpp>
#include <rocRoller/KernelGraph/KernelGraph.hpp>
#include <rocRoller/KernelGraph/Transforms/UpdateParameters.hpp>
#include <rocRoller/Operations/Command.hpp>

#include <common/CommonGraphs.hpp>

using namespace rocRoller;
using namespace rocRoller::KernelGraph;
using namespace rocRoller::KernelGraph::CoordinateGraph;
using namespace rocRoller::KernelGraph::ControlGraph;

TEST_CASE("SetUserSize for GEMM with different transposes", "[kernel-graph]")
{
    auto transposeConfig = GENERATE(values<std::pair<std::string, std::string>>(
        {{"N", "N"}, {"N", "T"}, {"T", "N"}, {"T", "T"}}));

    std::string transposeA = transposeConfig.first;
    std::string transposeB = transposeConfig.second;

    DYNAMIC_SECTION("transpose A=" << transposeA << " B=" << transposeB)
    {
        auto example = rocRollerTest::Graphs::GEMM(DataType::Half);
        example.setTranspose(transposeA, transposeB);

        auto command = example.getCommand();
        auto kgraph  = example.getKernelGraph();
        auto params  = example.getCommandParameters();

        // Apply UpdateParameters which includes SetUserSizeVisitor
        kgraph = kgraph.transform(std::make_shared<UpdateParameters>(params));

        // Find all User dimensions in the coordinate graph
        auto userTags = kgraph.coordinates.getNodes<User>().to<std::vector>();
        REQUIRE_FALSE(userTags.empty());

        // Track users that have sizes set
        int usersWithSize = 0;

        // Check each User dimension
        for(auto userTag : userTags)
        {
            auto user = kgraph.coordinates.get<User>(userTag);
            REQUIRE(user);

            // Skip Users without a defined size (e.g., scalars)
            if(!user->size)
                continue;

            usersWithSize++;
        }

        // Verify at least the main tensors (A, B, C/D) have sizes
        CHECK(usersWithSize == 4);
    }
}

TEST_CASE("SetUserSize for scaled GEMM", "[kernel-graph][update-parameters]")
{
    auto example = rocRollerTest::Graphs::GEMM(DataType::FP4);

    // Configure scaling
    example.setScaling(Operations::ScaleMode::Separate,
                       Operations::ScaleMode::Separate,
                       DataType::E8M0,
                       DataType::E8M0,
                       32);

    example.setTileSize(64, 64, 16);
    example.setMFMA(32, 32, 8, 1);
    example.setUseLDS(true, false, false);

    auto command = example.getCommand();
    auto kgraph  = example.getKernelGraph();
    auto params  = example.getCommandParameters();

    // Apply UpdateParameters which includes SetUserSizeVisitor
    kgraph = kgraph.transform(std::make_shared<UpdateParameters>(params));

    // Find all User dimensions in the coordinate graph
    auto userTags = kgraph.coordinates.getNodes<User>().to<std::vector>();
    REQUIRE_FALSE(userTags.empty());

    // Track users that have sizes set
    int usersWithSize = 0;

    // Check each User dimension
    for(auto userTag : userTags)
    {
        auto user = kgraph.coordinates.get<User>(userTag);
        REQUIRE(user);

        // Skip Users without a defined size (e.g., scalars)
        if(!user->size)
            continue;

        usersWithSize++;
    }

    // Verify at least the main tensors (A, B, C/D) have sizes
    CHECK(usersWithSize == 6);
}

TEST_CASE("SetUserSize for manually constructed load graph", "[kernel-graph][update-parameters]")
{
    rocRoller::KernelGraph::KernelGraph kgraph;

    auto kernel = kgraph.control.addElement(Kernel());

    // Create User for input tensor
    auto user = kgraph.coordinates.addElement(User({}, "input"));

    // Create Command and CommandArguments for dynamic sizes
    auto command    = std::make_shared<Command>();
    auto sizeMTag   = command->allocateTag();
    auto sizeKTag   = command->allocateTag();
    auto strideMTag = command->allocateTag();

    auto sizeMArg   = command->allocateArgument(VariableType{DataType::UInt64},
                                              sizeMTag,
                                              ArgumentType::Value,
                                              DataDirection::ReadOnly,
                                              "sizeM");
    auto sizeKArg   = command->allocateArgument(VariableType{DataType::UInt64},
                                              sizeKTag,
                                              ArgumentType::Value,
                                              DataDirection::ReadOnly,
                                              "sizeK");
    auto strideMArg = command->allocateArgument(VariableType{DataType::UInt64},
                                                strideMTag,
                                                ArgumentType::Value,
                                                DataDirection::ReadOnly,
                                                "strideM");

    // Create subdimensions with dynamic sizes using CommandArgumentPtr
    auto sizeM = std::make_shared<Expression::Expression>(sizeMArg);
    auto sizeK = std::make_shared<Expression::Expression>(sizeKArg);
    auto strideM
        = std::make_shared<Expression::Expression>(strideMArg); // Row-major: stride of M is K
    auto strideK = Expression::literal(1u);

    auto subDim0 = kgraph.coordinates.addElement(SubDimension(0, sizeM, strideM));
    auto subDim1 = kgraph.coordinates.addElement(SubDimension(1, sizeK, strideK));

    // Create MacroTile
    auto tileTag = kgraph.coordinates.addElement(MacroTile());

    // Build load pattern: User -> Split -> SubDims -> ConstructMacroTile -> MacroTile
    kgraph.coordinates.addElement(Split(), {user}, {subDim0, subDim1});
    kgraph.coordinates.addElement(ConstructMacroTile(), {subDim0, subDim1}, {tileTag});
    kgraph.coordinates.addElement(DataFlow(), {user}, {tileTag});

    // Set MacroTile properties
    auto tile       = *kgraph.coordinates.get<MacroTile>(tileTag);
    tile.layoutType = LayoutType::MATRIX_A;
    tile.rank       = 2;
    tile.memoryType = MemoryType::WAVE;
    kgraph.coordinates.setElement(tileTag, tile);

    // Create load operation
    auto load = kgraph.control.addElement(LoadTiled(DataType::Float));
    kgraph.control.addElement(Body(), {kernel}, {load});
    kgraph.mapper.connect<User>(load, user);
    kgraph.mapper.connect<MacroTile>(load, tileTag);

    // Create parameters
    // Row-major layout: rightmost dimension (K) is contiguous, so transposeMemoryAccess = true
    auto params = std::make_shared<CommandParameters>();
    params->transposeMemoryAccess.set(LayoutType::MATRIX_A, true);

    // Verify User.size is null before transform
    CHECK(!kgraph.coordinates.get<User>(user)->size);

    // Apply UpdateParameters
    kgraph = kgraph.transform(std::make_shared<UpdateParameters>(params));

    // Verify User.size is set after transform
    auto userAfter = kgraph.coordinates.get<User>(user);
    REQUIRE(userAfter);
    REQUIRE(userAfter->size);

    // For row-major layout, User.size should be stride_M * size_M
    // where stride_M is represented by strideMArg (which equals sizeK in row-major)
    // So User.size = strideM * sizeM
    auto expectedSize = strideM * sizeM;
    CHECK(Expression::identical(userAfter->size, expectedSize));
}

TEST_CASE("SetUserSize for manually constructed store graph", "[kernel-graph][update-parameters]")
{
    rocRoller::KernelGraph::KernelGraph kgraph;

    auto kernel = kgraph.control.addElement(Kernel());

    // Create MacroTile first (for store pattern)
    auto tileTag = kgraph.coordinates.addElement(MacroTile());

    // Create Command and CommandArguments for dynamic sizes
    auto command    = std::make_shared<Command>();
    auto sizeMTag   = command->allocateTag();
    auto sizeNTag   = command->allocateTag();
    auto strideMTag = command->allocateTag();

    auto sizeMArg   = command->allocateArgument(VariableType{DataType::UInt64},
                                              sizeMTag,
                                              ArgumentType::Value,
                                              DataDirection::ReadOnly,
                                              "sizeM");
    auto sizeNArg   = command->allocateArgument(VariableType{DataType::UInt64},
                                              sizeNTag,
                                              ArgumentType::Value,
                                              DataDirection::ReadOnly,
                                              "sizeN");
    auto strideMArg = command->allocateArgument(VariableType{DataType::UInt64},
                                                strideMTag,
                                                ArgumentType::Value,
                                                DataDirection::ReadOnly,
                                                "strideM");

    // Create subdimensions with dynamic sizes using CommandArgumentPtr
    auto sizeM = std::make_shared<Expression::Expression>(sizeMArg);
    auto sizeN = std::make_shared<Expression::Expression>(sizeNArg);
    auto strideM
        = std::make_shared<Expression::Expression>(strideMArg); // Row-major: stride of M is N
    auto strideN = Expression::literal(1u);

    auto subDim0 = kgraph.coordinates.addElement(SubDimension(0, sizeM, strideM));
    auto subDim1 = kgraph.coordinates.addElement(SubDimension(1, sizeN, strideN));

    // Create User for output tensor
    auto user = kgraph.coordinates.addElement(User({}, "output"));

    // Build store pattern: MacroTile -> DestructMacroTile -> SubDims -> Join -> User
    kgraph.coordinates.addElement(DestructMacroTile(), {tileTag}, {subDim0, subDim1});
    kgraph.coordinates.addElement(Join(), {subDim0, subDim1}, {user});
    kgraph.coordinates.addElement(DataFlow(), {tileTag}, {user});

    // Set MacroTile properties
    auto tile       = *kgraph.coordinates.get<MacroTile>(tileTag);
    tile.layoutType = LayoutType::MATRIX_ACCUMULATOR;
    tile.rank       = 2;
    tile.memoryType = MemoryType::WAVE;
    kgraph.coordinates.setElement(tileTag, tile);

    // Create store operation
    auto store = kgraph.control.addElement(StoreTiled(DataType::Float));
    kgraph.control.addElement(Body(), {kernel}, {store});
    kgraph.mapper.connect<User>(store, user);
    kgraph.mapper.connect<MacroTile>(store, tileTag);

    // Create parameters
    // Row-major layout: rightmost dimension (N) is contiguous, so transposeMemoryAccess = true
    auto params = std::make_shared<CommandParameters>();
    params->transposeMemoryAccess.set(LayoutType::MATRIX_ACCUMULATOR, true);

    // Verify User.size is null before transform
    CHECK(!kgraph.coordinates.get<User>(user)->size);

    // Apply UpdateParameters
    kgraph = kgraph.transform(std::make_shared<UpdateParameters>(params));

    // Verify User.size is set after transform
    auto userAfter = kgraph.coordinates.get<User>(user);
    REQUIRE(userAfter);
    REQUIRE(userAfter->size);

    // For row-major layout, User.size should be stride_M * size_M
    // where stride_M is represented by strideMArg (which equals sizeN in row-major)
    // So User.size = strideM * sizeM
    auto expectedSize = strideM * sizeM;
    CHECK(Expression::identical(userAfter->size, expectedSize));
}

TEST_CASE("SetUserSize with column-major layout", "[kernel-graph][update-parameters]")
{
    rocRoller::KernelGraph::KernelGraph kgraph;

    auto kernel = kgraph.control.addElement(Kernel());

    // Create User for input tensor
    auto user = kgraph.coordinates.addElement(User({}, "input"));

    // Create Command and CommandArguments for dynamic sizes
    auto command    = std::make_shared<Command>();
    auto sizeMTag   = command->allocateTag();
    auto sizeKTag   = command->allocateTag();
    auto strideKTag = command->allocateTag();

    auto sizeMArg   = command->allocateArgument(VariableType{DataType::UInt64},
                                              sizeMTag,
                                              ArgumentType::Value,
                                              DataDirection::ReadOnly,
                                              "sizeM");
    auto sizeKArg   = command->allocateArgument(VariableType{DataType::UInt64},
                                              sizeKTag,
                                              ArgumentType::Value,
                                              DataDirection::ReadOnly,
                                              "sizeK");
    auto strideKArg = command->allocateArgument(VariableType{DataType::UInt64},
                                                strideKTag,
                                                ArgumentType::Value,
                                                DataDirection::ReadOnly,
                                                "strideK");

    // Create subdimensions with dynamic sizes using CommandArgumentPtr for column-major layout
    auto sizeM   = std::make_shared<Expression::Expression>(sizeMArg);
    auto sizeK   = std::make_shared<Expression::Expression>(sizeKArg);
    auto strideM = Expression::literal(1u); // Column-major: stride of M is 1
    auto strideK
        = std::make_shared<Expression::Expression>(strideKArg); // Column-major: stride of K is M

    auto subDim0 = kgraph.coordinates.addElement(SubDimension(0, sizeM, strideM));
    auto subDim1 = kgraph.coordinates.addElement(SubDimension(1, sizeK, strideK));

    // Create MacroTile
    auto tileTag = kgraph.coordinates.addElement(MacroTile());

    // Build load pattern
    kgraph.coordinates.addElement(Split(), {user}, {subDim0, subDim1});
    kgraph.coordinates.addElement(ConstructMacroTile(), {subDim0, subDim1}, {tileTag});
    kgraph.coordinates.addElement(DataFlow(), {user}, {tileTag});

    // Set MacroTile properties
    auto tile       = *kgraph.coordinates.get<MacroTile>(tileTag);
    tile.layoutType = LayoutType::MATRIX_B;
    tile.rank       = 2;
    tile.memoryType = MemoryType::WAVE;
    kgraph.coordinates.setElement(tileTag, tile);

    // Create load operation
    auto load = kgraph.control.addElement(LoadTiled(DataType::Float));
    kgraph.control.addElement(Body(), {kernel}, {load});
    kgraph.mapper.connect<User>(load, user);
    kgraph.mapper.connect<MacroTile>(load, tileTag);

    // Create parameters with column-major (leftmost fastest)
    // Column-major layout: leftmost dimension (M) is contiguous, so transposeMemoryAccess = false
    auto params = std::make_shared<CommandParameters>();
    params->transposeMemoryAccess.set(LayoutType::MATRIX_B, false);

    // Verify User.size is null before transform
    CHECK(!kgraph.coordinates.get<User>(user)->size);

    // Apply UpdateParameters
    kgraph = kgraph.transform(std::make_shared<UpdateParameters>(params));

    // Verify User.size is set after transform
    auto userAfter = kgraph.coordinates.get<User>(user);
    REQUIRE(userAfter);
    REQUIRE(userAfter->size);

    // For column-major layout, User.size should be stride_K * size_K
    // where stride_K is represented by strideKArg (which equals sizeM in column-major)
    // So User.size = strideK * sizeK
    auto expectedSize = strideK * sizeK;
    CHECK(Expression::identical(userAfter->size, expectedSize));
}
