// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <Tensile/Tensile.hpp>
#ifdef TENSILE_YAML
#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/Serialization.hpp>
#include <llvm/Support/YAMLParser.h>
#include <llvm/Support/YAMLTraits.h>
#else
#include <Tensile/msgpack/MessagePack.hpp>
#endif

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

#ifdef TENSILE_YAML
namespace
{
    // LLVM YAML's document operators accept mappings and sequences. Wrap
    // scalar fixture values so LLVM also handles their quoting and parsing.
    template <typename T>
    struct YamlScalar
    {
        T value;
    };
}

namespace llvm::yaml
{
    template <typename T>
    struct MappingTraits<YamlScalar<T>>
    {
        static void mapping(IO& io, YamlScalar<T>& scalar)
        {
            io.mapRequired("value", scalar.value);
        }
    };
}

namespace
{
    struct YamlValue
    {
        std::string text;

        template <typename T>
        T as() const
        {
            YamlScalar<T> scalar{};
            std::string document = "{value: " + text + "}";
            llvm::yaml::Input reader(document);
            reader >> scalar;
            if(reader.error())
                throw std::runtime_error(reader.error().message());
            return scalar.value;
        }
    };

    template <typename T>
    YamlValue yamlObject(const T& value)
    {
        YamlScalar<T> scalar{value};
        std::string text;
        {
            llvm::raw_string_ostream stream(text);
            llvm::yaml::Output writer(stream);
            writer << scalar;
        }
        llvm::SourceMgr source;
        llvm::yaml::Stream stream(text, source);
        auto* mapping = llvm::cast<llvm::yaml::MappingNode>(stream.begin()->getRoot());
        auto* node = llvm::cast<llvm::yaml::ScalarNode>(mapping->begin()->getValue());
        return {node->getRawValue().str()};
    }

    YamlValue yamlObject(const YamlValue& value)
    {
        return value;
    }

    YamlValue yamlObject(const std::map<std::string, YamlValue>& fields)
    {
        std::string text = "{";
        for(const auto& [key, value] : fields)
            text += yamlObject(key).text + ": " + value.text + ", ";
        return {text + "}"};
    }

    template <typename T>
    YamlValue yamlObject(const std::vector<T>& values)
    {
        std::string text = "[";
        for(const auto& value : values)
            text += yamlObject(value).text + ", ";
        return {text + "]"};
    }
}
#endif

namespace TensileLite::Serialization
{
    // Record production mapping output as editable fixture fields. MessagePack
    // has an input adapter only; both backends round-trip these fields through
    // their production readers without duplicating the policy serializer.
    struct PolicyMappingOutput
    {
#ifdef TENSILE_YAML
        using Object = YamlValue;
#else
        using Object = msgpack::object;
        msgpack::zone& zone;
#endif
        std::map<std::string, Object> fields;
        Object scalar;

        template <typename T>
        Object object(const T& value)
        {
#ifdef TENSILE_YAML
            return yamlObject(value);
#else
            return msgpack::object(value, zone);
#endif
        }

        PolicyMappingOutput nested()
        {
#ifdef TENSILE_YAML
            return {};
#else
            return {zone};
#endif
        }

        template <typename T>
        Object encode(T& value);

        template <typename T>
        Object encode(vector3<T>& value)
        {
            std::array<T, 3> elements{value.x, value.y, value.z};
            return encode(elements);
        }

        template <typename T, size_t N>
        Object encode(std::array<T, N>& values)
        {
            std::vector<Object> elements;
            for(auto& value : values)
                elements.push_back(encode(value));
            return object(elements);
        }

        template <typename T>
        Object encode(std::vector<T>& values)
        {
            std::vector<Object> elements;
            for(auto& value : values)
                elements.push_back(encode(value));
            return object(elements);
        }
    };

    template <>
    struct IOTraits<PolicyMappingOutput>
    {
        template <typename T>
        static void mapRequired(PolicyMappingOutput& io, const char* key, T& value)
        {
            io.fields[key] = io.encode(value);
        }

        template <typename T>
        static void mapOptional(PolicyMappingOutput& io, const char* key, T& value)
        {
            mapRequired(io, key, value);
        }

        static bool outputting(PolicyMappingOutput&)
        {
            return true;
        }
        static bool hasKey(PolicyMappingOutput& io, const char* key)
        {
            return io.fields.count(key) != 0;
        }

        template <typename T>
        static void enumCase(PolicyMappingOutput& io, T& value, const char* key, T expected)
        {
            if(value == expected)
                io.scalar = io.object(std::string(key));
        }
    };

    template <typename T>
    PolicyMappingOutput::Object PolicyMappingOutput::encode(T& value)
    {
        if constexpr(has_EmptyMappingTraits<T, PolicyMappingOutput>::value)
        {
            auto child = nested();
            MappingTraits<T, PolicyMappingOutput>::mapping(child, value);
            return object(child.fields);
        }
        else if constexpr(has_EnumTraits<T, PolicyMappingOutput>::value)
        {
            auto child = nested();
            EnumTraits<T, PolicyMappingOutput>::enumeration(child, value);
            return child.scalar;
        }
        else
        {
            return object(value);
        }
    }
}

namespace
{
    using namespace TensileLite;
    using ObjectMap = std::map<std::string, Serialization::PolicyMappingOutput::Object>;

    struct PolicyCase
    {
        const char* strategy;
        const char* assignment;
        int         legacyMode;
        int         legacyForceDP;
    };

    const std::array<PolicyCase, 5> policies{{
        {"None", "StaticGrid", 0, 0},
        {"DataParallel", "StaticGrid", 3, 1},
        {"StreamK", "StaticGrid", 3, 0},
        {"StreamK", "DynamicWorkQueue", 4, 0},
        {"StreamK", "Hybrid", 5, 0},
    }};

    TEST(PersistentExecutionPolicyTest, NonPersistentAssignmentsDoNotActivateSchedulingOrDataParallelArgsV1)
    {
        for(auto assignment : {WorkAssignment::StaticGrid,
                               WorkAssignment::DynamicWorkQueue,
                               WorkAssignment::Hybrid})
        {
            ContractionSolution solution;
            solution.sizeMapping.workAssignment = assignment;
            solution.internalArgsSupport.version = 3;
            EXPECT_NO_THROW(solution.validatePersistentLoopArgs());
            EXPECT_FALSE(solution.sizeMapping.hasStaticAssignment());
            EXPECT_FALSE(solution.sizeMapping.hasDynamicAssignment());
            EXPECT_FALSE(solution.sizeMapping.hasHybridAssignment());
            solution.internalArgsSupport.persistentLoopArgsVersion = 1;
            EXPECT_THROW(solution.validatePersistentLoopArgs(), std::runtime_error);
        }
    }

    class PersistentExecutionPolicySerializationTest : public ::testing::Test
    {
    protected:
#ifndef TENSILE_YAML
        msgpack::zone zone;
#endif
        ObjectMap     size;
        ObjectMap     internalArgs;
        ObjectMap     problem;
        ObjectMap     custom;

#ifdef TENSILE_YAML
        template <typename T>
        YamlValue object(const T& value)
        {
            return yamlObject(value);
        }

        std::vector<uint8_t> libraryBytes(const ObjectMap& library)
        {
            auto text = object(library).text;
            return {text.begin(), text.end()};
        }
#else
        template <typename T>
        msgpack::object object(const T& value)
        {
            return msgpack::object(value, zone);
        }

        template <typename T>
        void input(const ObjectMap& fields, T& value)
        {
            msgpack::sbuffer packed;
            msgpack::pack(packed, fields);
            auto unpacked = msgpack::unpack(packed.data(), packed.size());
            Serialization::MessagePackInput reader(unpacked.get());
            reader.input(value);
            if(!reader.error.empty())
                throw std::runtime_error(reader.error.front());
        }

        std::vector<uint8_t> libraryBytes(const ObjectMap& library)
        {
            msgpack::sbuffer packed;
            msgpack::pack(packed, library);
            return {packed.data(), packed.data() + packed.size()};
        }
#endif

        template <typename T>
        ObjectMap output(T& value)
        {
#ifdef TENSILE_YAML
            Serialization::PolicyMappingOutput writer{};
#else
            Serialization::PolicyMappingOutput writer{zone};
#endif
            Serialization::MappingTraits<T, Serialization::PolicyMappingOutput>::mapping(writer, value);
            return writer.fields;
        }

        SizeMapping readSize()
        {
#ifdef TENSILE_YAML
            // Keep decoding inside the public host API: the broad LLVM YAML
            // traits also instantiate unrelated private host registrations.
            return readSolution()->sizeMapping;
#else
            SizeMapping value{};
            input(size, value);
            return value;
#endif
        }

        std::shared_ptr<ContractionSolution> readSolution()
        {
            auto predicate
                = object(ObjectMap{{"type", object(Predicates::True<Hardware>::Type())}});
            ObjectMap        fields{{"name", object(std::string("legacy_solution_SK3_DP1"))},
                                    {"kernelName", object(std::string("prebuilt_kernel_SK3_DP1"))},
                                    {"index", object(17)},
                                    {"hardwarePredicate", predicate},
                                    {"problemPredicate", predicate},
                                    {"taskPredicate", predicate},
                                    {"debugKernel", object(false)},
                                    {"sizeMapping", object(size)},
                                    {"internalArgsSupport", object(internalArgs)},
                                    {"customKernel", object(custom)},
                                    {"problemType", object(problem)}};
            ObjectMap        library{{"solutions", object(std::vector<ObjectMap>{fields})},
                                     {"library",
                                      object(ObjectMap{{"type", object(std::string("Single"))},
                                                       {"index", object(17)}})}};
            auto bytes = libraryBytes(library);
            auto loaded = LoadLibraryData<ContractionProblemGemm, ContractionSolution>(bytes);
            auto master = std::dynamic_pointer_cast<MasterContractionLibrary>(loaded);
            if(!master)
                throw std::runtime_error("Persistent policy library failed to deserialize");
            return master->solutions.at(17);
        }

        void canonical(const PolicyCase& policy)
        {
            size["tileProcessingStrategy"] = object(std::string(policy.strategy));
            size["workAssignment"]         = object(std::string(policy.assignment));
        }

        void legacy(const PolicyCase& policy)
        {
            size.erase("tileProcessingStrategy");
            size.erase("workAssignment");
            size["streamK"]            = object(policy.legacyMode);
            size["streamKForceDPOnly"] = object(policy.legacyForceDP);
        }

        void expectPolicy(const SizeMapping& value, const PolicyCase& policy)
        {
            EXPECT_EQ(std::string(toString(value.tileProcessingStrategy)), policy.strategy);
            EXPECT_EQ(std::string(toString(value.workAssignment)), policy.assignment);
        }

        void SetUp() override
        {
            SizeMapping defaults{};
            defaults.waveNum           = 4;
            defaults.workGroupSize     = {256, 1, 1};
            defaults.threadTile        = {4, 4, 1};
            defaults.macroTile         = {64, 64, 1};
            defaults.matrixInstruction = {16, 16, 4, 1};
            defaults.waveGroup         = {2, 2};
            size                       = output(defaults);
            ContractionSolution::InternalArgsSupport args;
            internalArgs = output(args);
            ContractionSolution::ProblemType type;
            type.operationIdentifier = "Contraction_l_Ailk_Bljk_Cijk_Dijk";
            problem                  = output(type);
            CustomKernel kernel;
            custom = output(kernel);
        }
    };

    class PersistentExecutionPolicyRoundTripTest
        : public PersistentExecutionPolicySerializationTest,
          public ::testing::WithParamInterface<PolicyCase>
    {
    };

    TEST_P(PersistentExecutionPolicyRoundTripTest, CanonicalSelectorsRoundTrip)
    {
        canonical(GetParam());
        auto decoded = readSize();
        expectPolicy(decoded, GetParam());
        size = output(decoded);
        EXPECT_EQ(size.at("tileProcessingStrategy").as<std::string>(), GetParam().strategy);
        EXPECT_EQ(size.at("workAssignment").as<std::string>(), GetParam().assignment);
        EXPECT_EQ(size.count("streamK"), 0u);
        EXPECT_EQ(size.count("streamKForceDPOnly"), 0u);
        expectPolicy(readSize(), GetParam());
    }

    TEST_P(PersistentExecutionPolicyRoundTripTest, LegacySelectorsEmitCanonicalSelectors)
    {
        legacy(GetParam());
        auto decoded = readSize();
        expectPolicy(decoded, GetParam());
        size = output(decoded);
        EXPECT_EQ(size.count("streamK"), 0u);
        EXPECT_EQ(size.count("streamKForceDPOnly"), 0u);
        expectPolicy(readSize(), GetParam());
    }

    TEST_P(PersistentExecutionPolicyRoundTripTest, ConsistentMixedSelectorsAreAccepted)
    {
        legacy(GetParam());
        canonical(GetParam());
        expectPolicy(readSize(), GetParam());
    }

    INSTANTIATE_TEST_SUITE_P(SupportedPolicies,
                             PersistentExecutionPolicyRoundTripTest,
                             ::testing::ValuesIn(policies),
                             [](const ::testing::TestParamInfo<PolicyCase>& info) {
                                 return std::string(info.param.strategy) + info.param.assignment;
                             });

    TEST_F(PersistentExecutionPolicySerializationTest, OmittedSelectorsDefaultToNonPersistent)
    {
        size.erase("tileProcessingStrategy");
        size.erase("workAssignment");
        expectPolicy(readSize(), policies[0]);
    }

    TEST_F(PersistentExecutionPolicySerializationTest, ConflictingMixedSelectorsAreRejected)
    {
        for(const auto& old : policies)
            for(const auto& requested : policies)
            {
                if(&old == &requested)
                    continue;
                SCOPED_TRACE(std::string(old.strategy) + old.assignment + " to "
                             + requested.strategy + requested.assignment);
                legacy(old);
                canonical(requested);
                EXPECT_THROW(readSize(), std::runtime_error);
            }
    }

    TEST_F(PersistentExecutionPolicySerializationTest, InvalidLegacyModesAreRejected)
    {
        legacy(policies[0]);
        for(int mode : {-2, -1, 1, 2, 6})
        {
            SCOPED_TRACE(mode);
            size["streamK"] = object(mode);
            EXPECT_THROW(readSize(), std::runtime_error);
        }
    }

    TEST_F(PersistentExecutionPolicySerializationTest, ForceDPRequiresNonAtomicStaticStreamK)
    {
        for(int mode : {0, 4, 5})
        {
            SCOPED_TRACE(mode);
            legacy(policies[1]);
            size["streamK"] = object(mode);
            EXPECT_THROW(readSize(), std::runtime_error);
        }
        legacy(policies[1]);
        size.erase("streamK");
        EXPECT_THROW(readSize(), std::runtime_error);
        legacy(policies[1]);
        size["streamKAtomic"] = object(1);
        EXPECT_THROW(readSize(), std::runtime_error);
    }

    TEST_F(PersistentExecutionPolicySerializationTest, InvalidForceDPValuesAreRejected)
    {
        legacy(policies[2]);
        for(int force : {-1, 2})
        {
            size["streamKForceDPOnly"] = object(force);
            EXPECT_THROW(readSize(), std::runtime_error);
        }
    }

    TEST_F(PersistentExecutionPolicySerializationTest, DataParallelRequiresStaticAssignment)
    {
        canonical(policies[1]);
        for(auto assignment : {"DynamicWorkQueue", "Hybrid"})
        {
            size["workAssignment"] = object(std::string(assignment));
            EXPECT_THROW(readSize(), std::runtime_error);
        }
    }

    TEST_F(PersistentExecutionPolicySerializationTest, NonPersistentAssignmentsAreInactive)
    {
        for(bool withLegacy : {false, true})
            for(bool withStrategy : {false, true})
                for(auto assignment : {"StaticGrid", "DynamicWorkQueue", "Hybrid"})
                {
                    SCOPED_TRACE(assignment);
                    if(withLegacy)
                        legacy(policies[0]);
                    if(withStrategy)
                        canonical(policies[0]);
                    else
                        size.erase("tileProcessingStrategy");
                    size["workAssignment"] = object(std::string(assignment));
                    auto decoded = readSize();
                    expectPolicy(decoded, policies[0]);
                    EXPECT_FALSE(decoded.isPersistent());
                    EXPECT_FALSE(decoded.hasStaticAssignment());
                    EXPECT_FALSE(decoded.hasDynamicAssignment());
                    EXPECT_FALSE(decoded.hasHybridAssignment());
                    size = output(decoded);
                    expectPolicy(readSize(), policies[0]);
                }
    }

    TEST_F(PersistentExecutionPolicySerializationTest, InvalidInactiveAssignmentNamesAreRejected)
    {
        for(bool withLegacy : {false, true})
            for(auto invalid : {"", "Bogus", "hybrid"})
            {
                if(withLegacy)
                    legacy(policies[0]);
                canonical(policies[0]);
                size["workAssignment"] = object(std::string(invalid));
                EXPECT_THROW(readSize(), std::runtime_error);
            }
    }

    TEST_F(PersistentExecutionPolicySerializationTest, InvalidCanonicalNamesAreRejected)
    {
        for(auto invalid : {"", "Bogus", "streamk"})
        {
            canonical(policies[2]);
            size["tileProcessingStrategy"] = object(std::string(invalid));
            EXPECT_THROW(readSize(), std::runtime_error);
            canonical(policies[2]);
            size["workAssignment"] = object(std::string(invalid));
            EXPECT_THROW(readSize(), std::runtime_error);
        }
    }

    TEST_F(PersistentExecutionPolicySerializationTest, AtomicRequiresStreamK)
    {
        size["streamKAtomic"] = object(1);
        for(const auto& policy : policies)
        {
            canonical(policy);
            if(std::string(policy.strategy) == "StreamK")
                EXPECT_NO_THROW(readSize());
            else
                EXPECT_THROW(readSize(), std::runtime_error);
        }
        canonical(policies[2]);
        for(int atomic : {-1, 2})
        {
            size["streamKAtomic"] = object(atomic);
            EXPECT_THROW(readSize(), std::runtime_error);
        }
    }

    TEST_F(PersistentExecutionPolicySerializationTest, LegacyOffClearsInactiveAtomicOption)
    {
        legacy(policies[0]);
        size["streamKAtomic"] = object(1);
        auto decoded          = readSize();
        expectPolicy(decoded, policies[0]);
        EXPECT_EQ(decoded.streamKAtomic, 0);
    }

    TEST_F(PersistentExecutionPolicySerializationTest, MissingArgumentLayoutVersionDefaultsToZero)
    {
        internalArgs.erase("persistentLoopArgsVersion");
        for(const auto& policy : policies)
        {
            canonical(policy);
            EXPECT_EQ(readSolution()->internalArgsSupport.persistentLoopArgsVersion, 0);
        }
    }

    TEST_F(PersistentExecutionPolicySerializationTest, ArgumentLayoutV1RequiresDataParallel)
    {
        internalArgs["persistentLoopArgsVersion"] = object(1);
        internalArgs["version"]                   = object(3);
        for(const auto& policy : policies)
        {
            canonical(policy);
            if(std::string(policy.strategy) == "DataParallel")
                EXPECT_EQ(readSolution()->internalArgsSupport.persistentLoopArgsVersion, 1);
            else
                EXPECT_THROW(readSolution(), std::runtime_error);
        }
    }

    TEST_F(PersistentExecutionPolicySerializationTest, ArgumentLayoutV1RequiresOuterProtocolThree)
    {
        canonical(policies[1]);
        internalArgs["persistentLoopArgsVersion"] = object(1);
        for(int version : {0, 1, 2})
        {
            SCOPED_TRACE(version);
            internalArgs["version"] = object(version);
            EXPECT_THROW(readSolution(), std::runtime_error);
        }
        internalArgs["version"] = object(3);
        auto decoded = readSolution();
        EXPECT_EQ(decoded->internalArgsSupport.version, 3);
        EXPECT_EQ(decoded->internalArgsSupport.persistentLoopArgsVersion, 1);
    }

    TEST_F(PersistentExecutionPolicySerializationTest, UnknownArgumentLayoutVersionsAreRejected)
    {
        canonical(policies[1]);
        internalArgs["version"] = object(3);
        for(int version : {-1, 2, 10})
        {
            internalArgs["persistentLoopArgsVersion"] = object(version);
            EXPECT_THROW(readSolution(), std::runtime_error);
        }
    }

    TEST_F(PersistentExecutionPolicySerializationTest, PrebuiltLegacyMetadataIsPreserved)
    {
        legacy(policies[1]);
        internalArgs.erase("persistentLoopArgsVersion");
        internalArgs["version"]           = object(2);
        internalArgs["perTileExtraIters"] = object(true);
        internalArgs["useUniversalArgs"]  = object(false);
        CustomKernel descriptor;
        descriptor.name      = "prebuilt_kernel_SK3_DP1";
        descriptor.generated = true;
        descriptor.macrotile = {128, 64, 1};
        descriptor.threads   = {256, 1, 1};
        descriptor.grid
            = {CustomGridSize::StreamKWithBatch, CustomGridSize::One, CustomGridSize::One};
        descriptor.args = {{CustomArgType::uint32, CustomArgSemantic::ItersPerTile, 0, 0},
                           {CustomArgType::uint32, CustomArgSemantic::SKItersPerWG, 0, 0},
                           {CustomArgType::uint32, CustomArgSemantic::SKGrid, 0, 0},
                           {CustomArgType::uint32, CustomArgSemantic::SKTilesAndSplit, 0, 0},
                           {CustomArgType::address, CustomArgSemantic::AddressD, 8, 3}};
        custom          = output(descriptor);
        auto loaded     = readSolution();
        expectPolicy(loaded->sizeMapping, policies[1]);
        EXPECT_EQ(loaded->solutionName, "legacy_solution_SK3_DP1");
        EXPECT_EQ(loaded->kernelName, "prebuilt_kernel_SK3_DP1");
        EXPECT_EQ(loaded->index, 17);
        EXPECT_EQ(loaded->internalArgsSupport.version, 2);
        EXPECT_EQ(loaded->internalArgsSupport.persistentLoopArgsVersion, 0);
        EXPECT_TRUE(loaded->internalArgsSupport.perTileExtraIters);
        EXPECT_FALSE(loaded->internalArgsSupport.useUniversalArgs);
        EXPECT_EQ(loaded->customKernel.name, descriptor.name);
        EXPECT_TRUE(loaded->customKernel.generated);
        EXPECT_EQ(loaded->customKernel.grid.x, CustomGridSize::StreamKWithBatch);
        EXPECT_EQ(loaded->customKernel.grid.y, CustomGridSize::One);
        EXPECT_EQ(loaded->customKernel.grid.z, CustomGridSize::One);
        EXPECT_EQ(loaded->customKernel.macrotile.x, 128u);
        EXPECT_EQ(loaded->customKernel.macrotile.y, 64u);
        EXPECT_EQ(loaded->customKernel.threads.x, 256u);
        ASSERT_EQ(loaded->customKernel.args.size(), descriptor.args.size());
        for(size_t i = 0; i < descriptor.args.size(); ++i)
        {
            EXPECT_EQ(loaded->customKernel.args[i].type, descriptor.args[i].type);
            EXPECT_EQ(loaded->customKernel.args[i].semantic, descriptor.args[i].semantic);
            EXPECT_EQ(loaded->customKernel.args[i].padding, descriptor.args[i].padding);
            EXPECT_EQ(loaded->customKernel.args[i].index, descriptor.args[i].index);
        }
        auto emittedDescriptor = output(loaded->customKernel);
        auto grid              = emittedDescriptor.at("grid").as<std::vector<std::string>>();
        EXPECT_EQ(grid, (std::vector<std::string>{"StreamKWithBatch", "One", "One"}));
    }
}
