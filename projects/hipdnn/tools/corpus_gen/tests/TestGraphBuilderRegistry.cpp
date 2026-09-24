// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file TestGraphBuilderRegistry.cpp
 * @brief Covers metadata to a real graph (RFC 0019.13 §4.3.6).
 *
 * This is the join that makes an operation addable by writing a file: a parameter dictionary
 * on one side, a builder that already exists on the other, and a declaration in between. What
 * is asserted is that the declaration is actually obeyed — a graph built from a stale default
 * rather than from the point is the failure mode, and it produces a perfectly good graph.
 */

#include <gtest/gtest.h>

#include <hipdnn_corpus_gen/GraphBuilderRegistry.hpp>
#include <hipdnn_corpus_gen/ProblemSpace.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>
#include <optional>
#include <string>

namespace hipdnn_corpus_gen
{
namespace
{

OperationMetadata load(const std::string& json)
{
    auto parsed = parseOperationMetadata(nlohmann::json::parse(json));
    EXPECT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());
    return parsed.metadata.value_or(OperationMetadata{});
}

/// Matmul declared end to end: the smallest complete example of §4.
OperationMetadata matmulMetadata()
{
    return load(R"({
      "schema_version": "1.0",
      "operation": "matmul",
      "parameters": {
        "M":     { "type": "int64" },
        "N":     { "type": "int64" },
        "K":     { "type": "int64" },
        "dtype": { "type": "enum", "values": ["fp32", "fp16", "bf16"] }
      },
      "stratification_axis": "arithmetic_intensity",
      "regimes": {},
      "graph_builder": {
        "function": "matmul",
        "source": "hipdnn_test_sdk/utilities/FlatbufferGraphTestUtils.hpp",
        "arguments": [
          { "name": "aDims", "kind": "expr", "value": ["$q.M", "$q.K"] },
          { "name": "aStrides", "kind": "strides_of", "of": "aDims" },
          { "name": "bDims", "kind": "expr", "value": ["$q.K", "$q.N"] },
          { "name": "bStrides", "kind": "strides_of", "of": "bDims" },
          { "name": "cDims", "kind": "expr", "value": ["$q.M", "$q.N"] },
          { "name": "cStrides", "kind": "strides_of", "of": "cDims" },
          { "name": "dataType", "kind": "dtype_of", "source": "$q.dtype" }
        ]
      }
    })");
}

/// Reads the graph back, which is the only way to check the declaration was obeyed.
const hipdnn_flatbuffers_sdk::data_objects::Graph* asGraph(const GraphBytes& bytes)
{
    return hipdnn_flatbuffers_sdk::data_objects::GetGraph(bytes.data());
}

} // namespace

TEST(TestGraphBuilderRegistry, BuildsAGraphFromADeclarationAndAProblemPoint)
{
    const ProblemPoint point{{"M", int64_t{128}},
                             {"N", int64_t{256}},
                             {"K", int64_t{64}},
                             {"dtype", std::string("fp16")}};

    const auto built = buildGraphFor(matmulMetadata(), point);

    ASSERT_TRUE(built.ok()) << built.error;
    const auto* graph = asGraph(built.bytes);
    ASSERT_NE(graph, nullptr);
    ASSERT_NE(graph->tensors(), nullptr);
    EXPECT_GE(graph->tensors()->size(), 3U);
}

TEST(TestGraphBuilderRegistry, TheProblemPointReachesTheTensors)
{
    // The assertion the whole layer exists for. A builder called with its defaults produces a
    // graph that is valid, benchmarks fine, and has nothing to do with the corpus row naming
    // it -- so it is not enough that a graph came back; the extents have to be the ones asked
    // for.
    const ProblemPoint point{{"M", int64_t{128}},
                             {"N", int64_t{256}},
                             {"K", int64_t{64}},
                             {"dtype", std::string("fp16")}};

    const auto built = buildGraphFor(matmulMetadata(), point);
    ASSERT_TRUE(built.ok()) << built.error;

    bool sawA = false;
    for(const auto* tensor : *asGraph(built.bytes)->tensors())
    {
        const auto* extents = tensor->dims();
        if(extents != nullptr && extents->size() == 2 && extents->Get(0) == 128
           && extents->Get(1) == 64)
        {
            sawA = true;
            // dtype is a parameter of the problem, so it must arrive too -- not the builder's
            // FLOAT default.
            EXPECT_EQ(tensor->data_type(), hipdnn_flatbuffers_sdk::data_objects::DataType::HALF);
        }
    }
    EXPECT_TRUE(sawA) << "no tensor carried the declared M x K extents";
}

TEST(TestGraphBuilderRegistry, DtypeIsTakenFromTheProblemNotTheDefault)
{
    // Each declared dtype must reach the graph, or a corpus sweeping dtypes is really sweeping
    // one and mislabelling the rest.
    for(const auto& [name, expected] :
        std::vector<std::pair<std::string, hipdnn_flatbuffers_sdk::data_objects::DataType>>{
            {"fp32", hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT},
            {"fp16", hipdnn_flatbuffers_sdk::data_objects::DataType::HALF},
            {"bf16", hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16}})
    {
        const ProblemPoint point{
            {"M", int64_t{8}}, {"N", int64_t{8}}, {"K", int64_t{8}}, {"dtype", name}};

        const auto built = buildGraphFor(matmulMetadata(), point);
        ASSERT_TRUE(built.ok()) << built.error;
        EXPECT_EQ(asGraph(built.bytes)->tensors()->Get(0)->data_type(), expected) << name;
    }
}

TEST(TestGraphBuilderRegistry, ArgumentsAreMatchedByNameNotByPosition)
{
    // §4.2's own worked example declares dims before strides while the builder it names takes
    // (strides, dims, ...). Under positional dispatch that swaps a tensor's extents with its
    // strides -- which builds, benchmarks, and is a different problem. Reordering the
    // declaration must therefore change nothing.
    auto reordered = nlohmann::json::parse(R"({
      "schema_version": "1.0",
      "operation": "matmul",
      "parameters": { "M": { "type": "int64" }, "N": { "type": "int64" },
                      "K": { "type": "int64" },
                      "dtype": { "type": "enum", "values": ["fp32"] } },
      "stratification_axis": "arithmetic_intensity",
      "regimes": {},
      "graph_builder": {
        "function": "matmul",
        "source": "x.hpp",
        "arguments": [
          { "name": "dataType", "kind": "dtype_of", "source": "$q.dtype" },
          { "name": "cDims", "kind": "expr", "value": ["$q.M", "$q.N"] },
          { "name": "cStrides", "kind": "strides_of", "of": "cDims" },
          { "name": "aDims", "kind": "expr", "value": ["$q.M", "$q.K"] },
          { "name": "aStrides", "kind": "strides_of", "of": "aDims" },
          { "name": "bDims", "kind": "expr", "value": ["$q.K", "$q.N"] },
          { "name": "bStrides", "kind": "strides_of", "of": "bDims" }
        ]
      }
    })");

    const ProblemPoint point{
        {"M", int64_t{32}}, {"N", int64_t{16}}, {"K", int64_t{8}}, {"dtype", std::string("fp32")}};

    const auto shuffled = buildGraphFor(parseOperationMetadata(reordered).metadata.value(), point);
    const auto ordered = buildGraphFor(matmulMetadata(),
                                       ProblemPoint{{"M", int64_t{32}},
                                                    {"N", int64_t{16}},
                                                    {"K", int64_t{8}},
                                                    {"dtype", std::string("fp32")}});

    ASSERT_TRUE(shuffled.ok()) << shuffled.error;
    ASSERT_TRUE(ordered.ok()) << ordered.error;
    EXPECT_EQ(shuffled.bytes, ordered.bytes) << "declaration order changed the graph";
}

TEST(TestGraphBuilderRegistry, NamesTheBuilderItCannotFind)
{
    // §4.4 check 5. A metadata file naming an unregistered builder cannot produce a problem,
    // and the message has to say which name so the fix is obvious.
    auto unknown = nlohmann::json::parse(R"({
      "schema_version": "1.0", "operation": "x",
      "parameters": { "M": { "type": "int64" } },
      "stratification_axis": "working_set", "regimes": {},
      "graph_builder": { "function": "noSuchBuilder", "source": "x.hpp",
                         "arguments": [] }
    })");

    const auto built = buildGraphFor(parseOperationMetadata(unknown).metadata.value(),
                                     ProblemPoint{{"M", int64_t{4}}});

    EXPECT_FALSE(built.ok());
    EXPECT_NE(built.error.find("noSuchBuilder"), std::string::npos);
}

TEST(TestGraphBuilderRegistry, DeclinesAnIncompleteDeclaration)
{
    // An adapter that filled in what the metadata omitted would be the same failure the
    // resolver refuses, one layer down.
    auto partial = nlohmann::json::parse(R"({
      "schema_version": "1.0", "operation": "matmul",
      "parameters": { "M": { "type": "int64" }, "K": { "type": "int64" } },
      "stratification_axis": "arithmetic_intensity", "regimes": {},
      "graph_builder": { "function": "matmul", "source": "x.hpp",
        "arguments": [ { "name": "aDims", "kind": "expr", "value": ["$q.M", "$q.K"] } ] }
    })");

    const auto built = buildGraphFor(parseOperationMetadata(partial).metadata.value(),
                                     ProblemPoint{{"M", int64_t{4}}, {"K", int64_t{4}}});

    EXPECT_FALSE(built.ok());
    EXPECT_NE(built.error.find("bDims"), std::string::npos);
}

TEST(TestGraphBuilderRegistry, RegistersTheBuildersAMetadataFileMayName)
{
    const auto names = registeredBuilders();
    EXPECT_NE(std::find(names.begin(), names.end(), "convolutionForward"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "pointwiseBinary"), names.end());
}

TEST(TestGraphBuilderRegistry, EveryDataTypeTheBackendAcceptsCanBeNamed)
{
    // The backend converts eighteen types in both directions (DataTypeConversion.cpp); a
    // hand-written copy of that list here named ten, so a declaration asking for fp4_e2m1 or a
    // fnuz variant was refused exactly as a misspelling would be. A corpus cannot cover a dtype
    // it cannot spell, and nothing in the output distinguishes "unsupported" from "unnameable".
    const std::vector<std::string> supported{"float",
                                             "double",
                                             "half",
                                             "bfloat16",
                                             "int8",
                                             "uint8",
                                             "int32",
                                             "int64",
                                             "boolean",
                                             "fp8_e4m3",
                                             "fp8_e5m2",
                                             "fp8_e8m0",
                                             "fp8_e4m3_fnuz",
                                             "fp8_e5m2_fnuz",
                                             "fp4_e2m1",
                                             "fp6_e2m3",
                                             "fp6_e3m2",
                                             "int4"};

    for(const auto& name : supported)
    {
        EXPECT_TRUE(detail::dataTypeFor(name).has_value()) << name << " cannot be named";
    }

    // The runtime's spellings, which are what the declarations use and what the corpus records.
    EXPECT_EQ(detail::dataTypeFor("fp32"), detail::dataTypeFor("float"));
    EXPECT_EQ(detail::dataTypeFor("fp16"), detail::dataTypeFor("half"));
    EXPECT_EQ(detail::dataTypeFor("bf16"), detail::dataTypeFor("bfloat16"));
    EXPECT_EQ(detail::dataTypeFor("fp64"), detail::dataTypeFor("double"));

    // The numpy spellings are refused, not quietly accepted. Resolving them would build a valid
    // graph and then write `float32` into the corpus's `q.dtype` column, so the
    // `categorical_encoding` generated from that corpus would hold `float32` while the runtime
    // binds `fp32` -- a spelling that corpus never held, with no code, refused at scoring.
    // Failing here names the offending declaration instead.
    EXPECT_FALSE(detail::dataTypeFor("float32").has_value());
    EXPECT_FALSE(detail::dataTypeFor("float16").has_value());
    EXPECT_FALSE(detail::dataTypeFor("float64").has_value());

    // And an unknown name is still a refusal rather than an untyped tensor.
    EXPECT_FALSE(detail::dataTypeFor("float17").has_value());
    EXPECT_FALSE(detail::dataTypeFor("unset").has_value());
}

TEST(TestGraphBuilderRegistry, EveryShippedDeclarationNamesABuilderThatExists)
{
    // A declaration naming a builder nobody registered produces no graphs and no error visible
    // from a corpus run -- the operation just contributes nothing, which reads exactly like an
    // engine that serves nothing. One declaration named a function that had been deleted and
    // stayed that way, so this is a hard failure rather than a logged note.
    const auto names = registeredBuilders();

    int declarationsSeen = 0;
    for(const auto& file : std::filesystem::directory_iterator(HIPDNN_CORPUS_GEN_OPERATIONS_DIR))
    {
        if(file.path().string().find(".opmeta.json") == std::string::npos)
        {
            continue;
        }
        std::ifstream stream(file.path());
        ASSERT_TRUE(stream.good()) << file.path();

        const auto parsed = parseOperationMetadata(nlohmann::json::parse(stream));
        ASSERT_TRUE(parsed.ok()) << file.path().filename() << ": "
                                 << (parsed.errors.empty() ? "" : parsed.errors.front());

        const auto& function = parsed.metadata->graphBuilder.function;
        EXPECT_NE(std::find(names.begin(), names.end(), function), names.end())
            << parsed.metadata->operation << " names an unregistered builder: " << function;
        ++declarationsSeen;
    }
    EXPECT_GT(declarationsSeen, 0) << "no declarations found; the check passed vacuously";
}

TEST(TestGraphBuilderRegistry, TheShippedConvolutionMetadataBuildsRealGraphs)
{
    // The operation declared entirely in a file, exercised through the same path a corpus run
    // uses. This is what "adding an operation is a metadata file" has to mean in practice, and
    // it is checked against the shipped artifact rather than an inline copy so the file itself
    // cannot rot.
    std::ifstream file(HIPDNN_CORPUS_GEN_OPERATIONS_DIR "/conv_fwd.opmeta.json");
    ASSERT_TRUE(file.good()) << "conv_fwd.opmeta.json is not where the build says it is";

    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    ASSERT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());

    // ResNet50 conv1: 224x224x3 -> 112x112x64, 7x7 filter, stride 2, pad 3.
    const ProblemPoint conv1{{"N", int64_t{64}},
                             {"C", int64_t{3}},
                             {"K", int64_t{64}},
                             {"groups", int64_t{1}},
                             {"H", int64_t{224}},
                             {"W", int64_t{224}},
                             {"R", int64_t{7}},
                             {"S", int64_t{7}},
                             {"pad_h", int64_t{3}},
                             {"pad_w", int64_t{3}},
                             {"stride_h", int64_t{2}},
                             {"stride_w", int64_t{2}},
                             {"dilation_h", int64_t{1}},
                             {"dilation_w", int64_t{1}},
                             {"dtype", std::string("fp16")}};

    const auto built = buildGraphFor(*parsed.metadata, conv1);
    ASSERT_TRUE(built.ok()) << built.error;

    // The output extents must be the ones convolution defines, not the builder's defaults:
    // floor((224 + 6 - 6 - 1)/2) + 1 = 112.
    bool sawOutput = false;
    for(const auto* tensor : *asGraph(built.bytes)->tensors())
    {
        const auto* extents = tensor->dims();
        if(extents != nullptr && extents->size() == 4 && extents->Get(0) == 64
           && extents->Get(1) == 64 && extents->Get(2) == 112 && extents->Get(3) == 112)
        {
            sawOutput = true;
        }
    }
    EXPECT_TRUE(sawOutput) << "no tensor carried the derived 64x64x112x112 output extents";
}

TEST(TestGraphBuilderRegistry, TheShippedMetadataCoversEveryDeclaredDtype)
{
    std::ifstream file(HIPDNN_CORPUS_GEN_OPERATIONS_DIR "/conv_fwd.opmeta.json");
    ASSERT_TRUE(file.good());
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    ASSERT_TRUE(parsed.ok());

    const auto* dtype = parsed.metadata->find("dtype");
    ASSERT_NE(dtype, nullptr);
    for(const auto& value : dtype->enumerable())
    {
        ProblemPoint point{{"N", int64_t{1}},
                           {"C", int64_t{2}},
                           {"K", int64_t{2}},
                           {"groups", int64_t{1}},
                           {"H", int64_t{8}},
                           {"W", int64_t{8}},
                           {"R", int64_t{3}},
                           {"S", int64_t{3}},
                           {"pad_h", int64_t{0}},
                           {"pad_w", int64_t{0}},
                           {"stride_h", int64_t{1}},
                           {"stride_w", int64_t{1}},
                           {"dilation_h", int64_t{1}},
                           {"dilation_w", int64_t{1}}};
        point["dtype"] = value;

        const auto built = buildGraphFor(*parsed.metadata, point);
        EXPECT_TRUE(built.ok()) << std::get<std::string>(value) << ": " << built.error;
    }
}

TEST(TestGraphBuilderRegistry, TheShippedConvolutionAdmitsARealLayer)
{
    // The declaration's constraints must admit the convolutions the declaration's own regimes
    // describe. A relation that is too strong empties the corpus silently -- the generator
    // reports no problems, which reads as an engine that serves nothing.
    std::ifstream file(HIPDNN_CORPUS_GEN_OPERATIONS_DIR "/conv_fwd.opmeta.json");
    ASSERT_TRUE(file.good());
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    ASSERT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());

    const ProblemPoint resnetLayer3{{"N", int64_t{64}},
                                    {"C", int64_t{512}},
                                    {"K", int64_t{512}},
                                    {"groups", int64_t{1}},
                                    {"H", int64_t{28}},
                                    {"W", int64_t{28}},
                                    {"R", int64_t{3}},
                                    {"S", int64_t{3}},
                                    {"pad_h", int64_t{0}},
                                    {"pad_w", int64_t{0}},
                                    {"stride_h", int64_t{1}},
                                    {"stride_w", int64_t{1}},
                                    {"dilation_h", int64_t{1}},
                                    {"dilation_w", int64_t{1}},
                                    {"dtype", std::string("fp16")}};

    EXPECT_TRUE(detail::satisfiesConstraints(*parsed.metadata, resnetLayer3))
        << "the shipped declaration rejects a 3x3 filter on a 28x28 input";

    // And must reject one that genuinely does not fit.
    auto tooLarge = resnetLayer3;
    tooLarge["R"] = int64_t{64};
    EXPECT_FALSE(detail::satisfiesConstraints(*parsed.metadata, tooLarge));
}

TEST(TestGraphBuilderRegistry, EveryDeclaredParameterReachesTheGraph)
{
    // The contract that justifies owning these builders rather than borrowing the test SDK's.
    //
    // A fixture may accept a parameter and ignore it -- createValidLayernormFpropGraph takes
    // inputDataType and computeDataType and writes io=FLOAT, intermediate=HALF,
    // compute=BFLOAT16 regardless -- and no test notices, because a fixture's job is to yield
    // *a* valid graph. For a corpus it is fatal: the row records what was asked for, the
    // hardware ran something else, and nothing downstream can tell.
    //
    // So: perturb one declared parameter at a time and require the emitted bytes to change.
    // Mechanical, and it is the check that would have caught every instance of this class.
    for(const auto& entry : std::filesystem::directory_iterator(HIPDNN_CORPUS_GEN_OPERATIONS_DIR))
    {
        if(entry.path().string().find(".opmeta.") == std::string::npos)
        {
            continue;
        }
        std::ifstream file(entry.path());
        ASSERT_TRUE(file.good()) << entry.path();
        const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
        ASSERT_TRUE(parsed.ok()) << entry.path().filename().string() << ": "
                                 << (parsed.errors.empty() ? "" : parsed.errors.front());
        const auto& metadata = *parsed.metadata;

        // A baseline every operation can build: a modest extent everywhere, the first declared
        // value for each categorical. Validity is not required -- only that the bytes respond.
        ProblemPoint baseline;
        for(const auto& parameter : metadata.parameters)
        {
            if(parameter.type == ParameterType::ENUM)
            {
                baseline[parameter.name] = parameter.values.front();
            }
            else if(parameter.type == ParameterType::BOOL)
            {
                baseline[parameter.name] = false;
            }
            else
            {
                baseline[parameter.name] = int64_t{8};
            }
        }

        const auto reference = buildGraphFor(metadata, baseline);
        if(!reference.ok())
        {
            // No owned builder yet: reported by name so an unregistered operation is visible
            // rather than quietly exempt from the contract.
            GTEST_LOG_(INFO) << metadata.operation << ": " << reference.error;
            continue;
        }

        for(const auto& parameter : metadata.parameters)
        {
            auto perturbed = baseline;
            if(parameter.type == ParameterType::ENUM)
            {
                if(parameter.values.size() < 2)
                {
                    continue;
                }
                perturbed[parameter.name] = parameter.values.back();
            }
            else if(parameter.type == ParameterType::BOOL)
            {
                perturbed[parameter.name] = true;
            }
            else
            {
                perturbed[parameter.name] = int64_t{16};
            }

            const auto changed = buildGraphFor(metadata, perturbed);
            ASSERT_TRUE(changed.ok())
                << metadata.operation << "." << parameter.name << ": " << changed.error;
            EXPECT_NE(changed.bytes, reference.bytes)
                << metadata.operation << " declares '" << parameter.name
                << "' but changing it does not change the graph";
        }
    }
}

TEST(TestGraphBuilderRegistry, ADeclaredDtypeReachesTheGraphHeaderAndEveryTensor)
{
    // The byte-difference check above is necessary and not sufficient: it catches a parameter
    // ignored entirely, and misses one applied partially. That is exactly the LayerNorm
    // fixture's failure -- dtype reaches the tensors, so the bytes change, while the graph
    // header stays io=FLOAT, intermediate=HALF, compute=BFLOAT16 and the graph will not
    // deserialize. Verified by mutation: reintroducing that bug leaves the byte check green.
    //
    // So a dtype_of argument is followed to where it must arrive.
    for(const auto& entry : std::filesystem::directory_iterator(HIPDNN_CORPUS_GEN_OPERATIONS_DIR))
    {
        if(entry.path().string().find(".opmeta.") == std::string::npos)
        {
            continue;
        }
        std::ifstream file(entry.path());
        const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
        ASSERT_TRUE(parsed.ok());
        const auto& metadata = *parsed.metadata;

        const auto* dtype = metadata.find("dtype");
        if(dtype == nullptr || dtype->values.empty())
        {
            continue;
        }

        for(const auto& declared : dtype->values)
        {
            ProblemPoint point;
            for(const auto& parameter : metadata.parameters)
            {
                if(parameter.name == "dtype")
                {
                    point[parameter.name] = declared;
                }
                else if(parameter.type == ParameterType::ENUM)
                {
                    point[parameter.name] = parameter.values.front();
                }
                else if(parameter.type == ParameterType::BOOL)
                {
                    point[parameter.name] = false;
                }
                else
                {
                    point[parameter.name] = int64_t{8};
                }
            }

            const auto built = buildGraphFor(metadata, point);
            if(!built.ok())
            {
                continue; // no owned builder; reported by the byte-difference test
            }

            const auto expected = detail::dataTypeFor(declared);
            ASSERT_TRUE(expected.has_value()) << declared;

            const auto* graph = asGraph(built.bytes);
            ASSERT_NE(graph, nullptr);
            EXPECT_EQ(graph->io_data_type(), *expected)
                << metadata.operation << " with dtype=" << declared
                << ": graph io_data_type does not follow the declaration";
            // Compute type follows the declaration too, but the declaration may separate it
            // from the operands: fp16 storage with fp32 accumulate is the ordinary mixed
            // precision case, and MIOpen's convolution builder requires exactly that. So the
            // expectation is whatever `computeDataType` names, falling back to the operand type
            // when a declaration does not separate them.
            auto expectedCompute = expected;
            for(const auto& argument : metadata.graphBuilder.arguments)
            {
                if(argument.name == "computeDataType" && argument.constant.is_string())
                {
                    expectedCompute = detail::dataTypeFor(argument.constant.get<std::string>());
                }
            }
            ASSERT_TRUE(expectedCompute.has_value());
            EXPECT_EQ(graph->compute_data_type(), *expectedCompute)
                << metadata.operation << " with dtype=" << declared
                << ": graph compute_data_type does not follow the declaration";
        }
    }
}

namespace
{

/// SDPA declared with a causal-anchor axis, which is the only argument these three tests are
/// about. The geometry is deliberately Sq < Sk, because that is the only regime where the two
/// anchors mask different triangles and so the only regime where getting this wrong is a
/// different problem rather than the same one spelled twice.
OperationMetadata sdpaAlignmentMetadata()
{
    return load(R"({
      "schema_version": "1.0",
      "operation": "sdpa_fwd",
      "parameters": {
        "alignment": { "type": "enum", "values": ["top_left", "bottom_right"] }
      },
      "stratification_axis": "working_set",
      "regimes": {},
      "graph_builder": {
        "function": "sdpaForward",
        "source": "hipdnn_corpus_gen/GraphBuilders.hpp",
        "arguments": [
          { "name": "qDims", "kind": "constant", "constant": [1, 4, 128, 64] },
          { "name": "qStrides", "kind": "strides_of", "of": "qDims" },
          { "name": "kDims", "kind": "constant", "constant": [1, 4, 4096, 64] },
          { "name": "kStrides", "kind": "strides_of", "of": "kDims" },
          { "name": "vDims", "kind": "constant", "constant": [1, 4, 4096, 64] },
          { "name": "vStrides", "kind": "strides_of", "of": "vDims" },
          { "name": "oDims", "kind": "constant", "constant": [1, 4, 128, 64] },
          { "name": "oStrides", "kind": "strides_of", "of": "oDims" },
          { "name": "causalMask", "kind": "constant", "constant": true },
          { "name": "dataType", "kind": "constant", "constant": "bf16" },
          { "name": "diagonalAlignment", "kind": "direct", "source": "$q.alignment" }
        ]
      }
    })");
}

/// The anchor as the graph records it, read back from the SDPA node.
std::optional<hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment>
    recordedAlignment(const GraphBytes& bytes)
{
    const auto* graph = asGraph(bytes);
    if(graph == nullptr || graph->nodes() == nullptr)
    {
        return std::nullopt;
    }
    for(const auto* node : *graph->nodes())
    {
        const auto* attributes = node->attributes_as_SdpaAttributes();
        if(attributes != nullptr)
        {
            return attributes->diagonal_alignment();
        }
    }
    return std::nullopt;
}

} // namespace

TEST(TestGraphBuilderRegistry, TheCausalAnchorFollowsTheDeclarationNotTheSchemaDefault)
{
    using hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment;

    for(const auto& [declared, expected] :
        std::vector<std::pair<std::string, DiagonalAlignment>>{
            {"top_left", DiagonalAlignment::TOP_LEFT},
            {"bottom_right", DiagonalAlignment::BOTTOM_RIGHT}})
    {
        const auto built = buildGraphFor(sdpaAlignmentMetadata(), ProblemPoint{{"alignment", declared}});

        ASSERT_TRUE(built.ok()) << declared << ": " << built.error;
        const auto recorded = recordedAlignment(built.bytes);
        ASSERT_TRUE(recorded.has_value()) << declared << ": no SDPA node in the graph";
        EXPECT_EQ(*recorded, expected) << declared;
    }
}

TEST(TestGraphBuilderRegistry, AnUndeclaredAnchorIsRefusedRatherThanReadAsTopLeft)
{
    // The reason this is not a default: at Sq < Sk the two anchors are different work, so a
    // spelling nobody defined, read as TOP_LEFT, would build, benchmark and record a corpus of
    // the wrong problem with no line anywhere saying so.
    const auto built
        = buildGraphFor(sdpaAlignmentMetadata(), ProblemPoint{{"alignment", std::string("middle")}});

    EXPECT_FALSE(built.ok());
    EXPECT_NE(built.error.find("diagonalAlignment"), std::string::npos) << built.error;
}

TEST(TestGraphBuilderRegistry, ADeclarationThatNeverHeardOfTheAnchorStillBuildsWhatItAlwaysBuilt)
{
    // TOP_LEFT is the schema default, so an unset argument must write the bytes a declaration
    // predating this axis wrote -- otherwise every corpus id in flight would move.
    auto metadata  = sdpaAlignmentMetadata();
    auto& arguments = metadata.graphBuilder.arguments;
    arguments.erase(std::remove_if(arguments.begin(),
                                   arguments.end(),
                                   [](const BuilderArgument& argument) {
                                       return argument.name == "diagonalAlignment";
                                   }),
                    arguments.end());

    const auto without = buildGraphFor(metadata, ProblemPoint{});
    ASSERT_TRUE(without.ok()) << without.error;

    const auto with = buildGraphFor(sdpaAlignmentMetadata(),
                                    ProblemPoint{{"alignment", std::string("top_left")}});
    ASSERT_TRUE(with.ok()) << with.error;

    EXPECT_EQ(without.bytes, with.bytes);
}

TEST(TestGraphBuilderRegistry, TheShippedSdpaMetadataBuildsGraphsTheEnginesAccept)
{
    // Three properties no device-free check would notice and each of which, on hardware,
    // empties a live-engine corpus completely -- found that way, one at a time:
    //  - fp32 compute: hip-kernel-provider's SdpaFwdPlanBuilder refuses any other
    //    ("Compute data type must be FLOAT"), and the ingestor engines decline at graph_match;
    //  - BSHD strides: rocKE's dense kernels take no stride kernargs, so its matcher declines
    //    the row-major BHSD layout `strides_of` would give;
    //  - an explicit attn_scale_value: rocKE requires one rather than guessing.
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    std::ifstream file(HIPDNN_CORPUS_GEN_OPERATIONS_DIR "/sdpa_fwd.opmeta.json");
    ASSERT_TRUE(file.good()) << "sdpa_fwd.opmeta.json is not where the build says it is";
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    ASSERT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());

    // GQA and Sq != Sk, so a stride that used the wrong head count or sequence length shows.
    const ProblemPoint point{{"batch", int64_t{2}},
                             {"heads", int64_t{8}},
                             {"heads_kv", int64_t{2}},
                             {"seqlen_q", int64_t{16}},
                             {"seqlen_k", int64_t{32}},
                             {"head_dim", int64_t{64}},
                             {"is_causal", false},
                             {"alignment", std::string("top_left")},
                             {"dtype", std::string("fp16")}};
    const auto built = buildGraphFor(*parsed.metadata, point);
    ASSERT_TRUE(built.ok()) << built.error;
    const auto* graph = asGraph(built.bytes);
    ASSERT_NE(graph, nullptr);

    EXPECT_EQ(graph->io_data_type(), DataType::HALF);
    EXPECT_EQ(graph->compute_data_type(), DataType::FLOAT);
    EXPECT_EQ(graph->intermediate_data_type(), DataType::FLOAT);

    ASSERT_NE(graph->nodes(), nullptr);
    ASSERT_EQ(graph->nodes()->size(), 1U);
    const auto* node = graph->nodes()->Get(0);
    EXPECT_EQ(node->compute_data_type(), DataType::FLOAT);
    const auto* attributes = node->attributes_as_SdpaAttributes();
    ASSERT_NE(attributes, nullptr);
    ASSERT_TRUE(attributes->attn_scale_value().has_value());
    EXPECT_FLOAT_EQ(attributes->attn_scale_value().value(), 0.125F); // 1/sqrt(64)

    // (B, H, S, D) dims, token-major: [S*H*D, D, H*D, 1].
    const std::map<std::string, std::vector<int64_t>> expected{
        {"q", {16 * 8 * 64, 64, 8 * 64, 1}},
        {"k", {32 * 2 * 64, 64, 2 * 64, 1}},
        {"v", {32 * 2 * 64, 64, 2 * 64, 1}},
        {"o", {16 * 8 * 64, 64, 8 * 64, 1}}};
    size_t seen = 0;
    for(const auto* tensor : *graph->tensors())
    {
        const auto found = expected.find(tensor->name()->str());
        if(found == expected.end())
        {
            continue;
        }
        ++seen;
        const std::vector<int64_t> strides(tensor->strides()->begin(), tensor->strides()->end());
        EXPECT_EQ(strides, found->second) << tensor->name()->str();
    }
    EXPECT_EQ(seen, expected.size());
}

TEST(TestGraphBuilderRegistry, ACausalSdpaGraphSaysCausalWithBoundsNotTheDeprecatedFlag)
{
    // Providers give causal_mask precedence over the bounds and read it as top-left whatever
    // diagonal_alignment says, so a bottom-right problem built with the flag is top-left -- and
    // AITER on gfx942, whose causal kernels are all bottom-right, declined every one.
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    std::ifstream file(HIPDNN_CORPUS_GEN_OPERATIONS_DIR "/sdpa_fwd.opmeta.json");
    ASSERT_TRUE(file.good());
    const auto parsed = parseOperationMetadata(nlohmann::json::parse(file));
    ASSERT_TRUE(parsed.ok()) << (parsed.errors.empty() ? "" : parsed.errors.front());

    for(const auto& [anchor, expected] :
        std::vector<std::pair<std::string, DiagonalAlignment>>{
            {"top_left", DiagonalAlignment::TOP_LEFT},
            {"bottom_right", DiagonalAlignment::BOTTOM_RIGHT}})
    {
        const ProblemPoint point{{"batch", int64_t{1}},
                                 {"heads", int64_t{8}},
                                 {"heads_kv", int64_t{8}},
                                 {"seqlen_q", int64_t{16}},
                                 {"seqlen_k", int64_t{32}},
                                 {"head_dim", int64_t{128}},
                                 {"is_causal", true},
                                 {"alignment", anchor},
                                 {"dtype", std::string("bf16")}};
        const auto built = buildGraphFor(*parsed.metadata, point);
        ASSERT_TRUE(built.ok()) << anchor << ": " << built.error;
        const auto* attributes
            = asGraph(built.bytes)->nodes()->Get(0)->attributes_as_SdpaAttributes();
        ASSERT_NE(attributes, nullptr);
        EXPECT_FALSE(attributes->causal_mask()) << anchor;
        EXPECT_FALSE(attributes->causal_mask_bottom_right()) << anchor;
        EXPECT_FALSE(attributes->left_bound().has_value()) << anchor;
        ASSERT_TRUE(attributes->right_bound().has_value()) << anchor;
        EXPECT_EQ(attributes->right_bound().value(), 0) << anchor;
        EXPECT_EQ(attributes->diagonal_alignment(), expected) << anchor;
    }
}

} // namespace hipdnn_corpus_gen
