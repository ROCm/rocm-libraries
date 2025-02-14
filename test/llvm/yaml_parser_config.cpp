/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include <vector>

#include <llvm/ObjectYAML/YAML.h>

#include <hiptensor/hiptensor.hpp>

#include "01_contraction/contraction_test_params.hpp"
#include "02_permutation/permutation_test_params.hpp"
#include "03_reduction/reduction_test_params.hpp"
#include "common.hpp"
#include "yaml_parser_impl.hpp"

// Fwd declare NoneType
namespace hiptensor
{
    struct NoneType;
    static constexpr hipDataType NONE_TYPE = (hipDataType)31;
}

// namespace hiptensor
// {
//     ///
//     /// Generalized params for contraction tests
//     ///
//     struct ContractionTestParams
//     {

//         using TestDataTypeT = std::vector<hipDataType>;

//         using AlgorithmT = hiptensorAlgo_t;
//         using OperatorT = hiptensorOperator_t;
//         using WorkSizePrefT = hiptensorWorksizePreference_t;
//         using LogLevelT = hiptensorLogLevel_t;

//         using LengthsT     = std::vector<std::size_t>;
//         using StridesT     = std::vector<std::size_t>;
//         using AlphaT       = double;
//         using BetaT        = double;

//         //Data types of input and output tensors
//         std::vector<TestDataTypeT> mDataTypes;
//         std::vector<AlgorithmT> mAlgorithms;
//         std::vector<OperatorT> mOperators;
//         std::vector<WorkSizePrefT> mWorkSizePrefs;
//         LogLevelT mLogLevelMask;
//         std::vector<LengthsT> mProblemLengths;
//         std::vector<StridesT> mProblemStrides;
//         std::vector<AlphaT> mAlphas;
//         std::vector<BetaT> mBetas;
//     };
// }

// Make custom types for Alpha and Beta types.
// We want to differentiate their treatment
// when vectorized. Under the hood they are
// doubles.
LLVM_YAML_STRONG_TYPEDEF(double, AlphaT);
LLVM_YAML_STRONG_TYPEDEF(double, BetaT);

// Treatment of types as vector elements
// Flow sequence vector is inline comma-separated values [val0, val1, ...]
// Sequence vector is line-break separated values
// - val0
// - val1
// ...
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(hipDataType)
LLVM_YAML_IS_SEQUENCE_VECTOR(hiptensorAlgo_t)
LLVM_YAML_IS_SEQUENCE_VECTOR(hiptensorOperator_t)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<hiptensorOperator_t>)
LLVM_YAML_IS_SEQUENCE_VECTOR(hiptensorWorksizePreference_t)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<hipDataType>)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::vector<std::size_t>)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<std::vector<std::size_t>>)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<std::vector<std::vector<std::size_t>>>)
LLVM_YAML_IS_FLOW_SEQUENCE_VECTOR(std::vector<int32_t>)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<std::vector<int32_t>>)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<std::vector<std::vector<int32_t>>>)
LLVM_YAML_IS_SEQUENCE_VECTOR(std::vector<double>)
LLVM_YAML_IS_SEQUENCE_VECTOR(AlphaT)
LLVM_YAML_IS_SEQUENCE_VECTOR(BetaT)

namespace llvm
{

    namespace yaml
    {
        ///
        // Enums encoding definitions
        ///

        template <>
        struct ScalarEnumerationTraits<hipDataType>
        {
            static void enumeration(IO& io, hipDataType& value)
            {
                io.enumCase(value, "HIP_R_16F", HIP_R_16F);
                io.enumCase(value, "HIP_R_16BF", HIP_R_16BF);
                io.enumCase(value, "HIP_R_32F", HIP_R_32F);
                io.enumCase(value, "HIP_R_64F", HIP_R_64F);
                io.enumCase(value, "HIP_C_32F", HIP_C_32F);
                io.enumCase(value, "HIP_C_64F", HIP_C_64F);
                io.enumCase(value, "NONE_TYPE", hiptensor::NONE_TYPE);
            }
        };

        template <>
        struct ScalarEnumerationTraits<hiptensorAlgo_t>
        {
            static void enumeration(IO& io, hiptensorAlgo_t& value)
            {
                io.enumCase(value, "HIPTENSOR_ALGO_ACTOR_CRITIC", HIPTENSOR_ALGO_ACTOR_CRITIC);
                io.enumCase(value, "HIPTENSOR_ALGO_DEFAULT", HIPTENSOR_ALGO_DEFAULT);
                io.enumCase(
                    value, "HIPTENSOR_ALGO_DEFAULT_PATIENT", HIPTENSOR_ALGO_DEFAULT_PATIENT);
            }
        };

        template <>
        struct ScalarEnumerationTraits<hiptensorOperator_t>
        {
            static void enumeration(IO& io, hiptensorOperator_t& value)
            {
                io.enumCase(value, "HIPTENSOR_OP_IDENTITY", HIPTENSOR_OP_IDENTITY);
                io.enumCase(value, "HIPTENSOR_OP_SQRT", HIPTENSOR_OP_SQRT);
                io.enumCase(value, "HIPTENSOR_OP_RELU", HIPTENSOR_OP_RELU);
                io.enumCase(value, "HIPTENSOR_OP_CONJ", HIPTENSOR_OP_CONJ);
                io.enumCase(value, "HIPTENSOR_OP_RCP", HIPTENSOR_OP_RCP);
                io.enumCase(value, "HIPTENSOR_OP_SIGMOID", HIPTENSOR_OP_SIGMOID);
                io.enumCase(value, "HIPTENSOR_OP_TANH", HIPTENSOR_OP_TANH);
                io.enumCase(value, "HIPTENSOR_OP_EXP", HIPTENSOR_OP_EXP);
                io.enumCase(value, "HIPTENSOR_OP_LOG", HIPTENSOR_OP_LOG);
                io.enumCase(value, "HIPTENSOR_OP_ABS", HIPTENSOR_OP_ABS);
                io.enumCase(value, "HIPTENSOR_OP_NEG", HIPTENSOR_OP_NEG);
                io.enumCase(value, "HIPTENSOR_OP_SIN", HIPTENSOR_OP_SIN);
                io.enumCase(value, "HIPTENSOR_OP_COS", HIPTENSOR_OP_COS);
                io.enumCase(value, "HIPTENSOR_OP_TAN", HIPTENSOR_OP_TAN);
                io.enumCase(value, "HIPTENSOR_OP_SINH", HIPTENSOR_OP_SINH);
                io.enumCase(value, "HIPTENSOR_OP_COSH", HIPTENSOR_OP_COSH);
                io.enumCase(value, "HIPTENSOR_OP_ASIN", HIPTENSOR_OP_ASIN);
                io.enumCase(value, "HIPTENSOR_OP_ACOS", HIPTENSOR_OP_ACOS);
                io.enumCase(value, "HIPTENSOR_OP_ATAN", HIPTENSOR_OP_ATAN);
                io.enumCase(value, "HIPTENSOR_OP_ASINH", HIPTENSOR_OP_ASINH);
                io.enumCase(value, "HIPTENSOR_OP_ACOSH", HIPTENSOR_OP_ACOSH);
                io.enumCase(value, "HIPTENSOR_OP_ATANH", HIPTENSOR_OP_ATANH);
                io.enumCase(value, "HIPTENSOR_OP_CEIL", HIPTENSOR_OP_CEIL);
                io.enumCase(value, "HIPTENSOR_OP_FLOOR", HIPTENSOR_OP_FLOOR);
                io.enumCase(value, "HIPTENSOR_OP_ADD", HIPTENSOR_OP_ADD);
                io.enumCase(value, "HIPTENSOR_OP_MUL", HIPTENSOR_OP_MUL);
                io.enumCase(value, "HIPTENSOR_OP_MIN", HIPTENSOR_OP_MIN);
                io.enumCase(value, "HIPTENSOR_OP_MAX", HIPTENSOR_OP_MAX);
                io.enumCase(value, "HIPTENSOR_OP_UNKNOWN", HIPTENSOR_OP_UNKNOWN);
            }
        };

        template <>
        struct ScalarEnumerationTraits<hiptensorWorksizePreference_t>
        {
            static void enumeration(IO& io, hiptensorWorksizePreference_t& value)
            {
                io.enumCase(value, "HIPTENSOR_WORKSPACE_MIN", HIPTENSOR_WORKSPACE_MIN);
                io.enumCase(
                    value, "HIPTENSOR_WORKSPACE_RECOMMENDED", HIPTENSOR_WORKSPACE_RECOMMENDED);
                io.enumCase(value, "HIPTENSOR_WORKSPACE_MAX", HIPTENSOR_WORKSPACE_MAX);
            }
        };

        ///
        // Bitfield for logging
        // Treating the hiptensorLogLevel as a bitfield, which is human readable.
        // Comma-separated values are inlined, but combined (|) into a final bit pattern.
        ///
        template <>
        struct ScalarBitSetTraits<hiptensorLogLevel_t>
        {
            static void bitset(IO& io, hiptensorLogLevel_t& value)
            {
                io.bitSetCase(value, "HIPTENSOR_LOG_LEVEL_OFF", HIPTENSOR_LOG_LEVEL_OFF);
                io.bitSetCase(value, "HIPTENSOR_LOG_LEVEL_ERROR", HIPTENSOR_LOG_LEVEL_ERROR);
                io.bitSetCase(
                    value, "HIPTENSOR_LOG_LEVEL_PERF_TRACE", HIPTENSOR_LOG_LEVEL_PERF_TRACE);
                io.bitSetCase(
                    value, "HIPTENSOR_LOG_LEVEL_PERF_HINT", HIPTENSOR_LOG_LEVEL_PERF_HINT);
                io.bitSetCase(value,
                              "HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE",
                              HIPTENSOR_LOG_LEVEL_HEURISTICS_TRACE);
                io.bitSetCase(
                    value, "HIPTENSOR_LOG_LEVEL_API_TRACE", HIPTENSOR_LOG_LEVEL_API_TRACE);
            }
        };

        ///
        // Define treatments of customized datatypes (passthrough to original types)
        ///

        template <>
        struct ScalarTraits<AlphaT> : public ScalarTraits<double>
        {
            using Base = ScalarTraits<double>;

            static void output(const AlphaT& value, void* v, llvm::raw_ostream& out)
            {
                Base::output(value, v, out);
            }

            static StringRef input(StringRef scalar, void* v, AlphaT& value)
            {
                return Base::input(scalar, v, (double&)(value));
            }
        };

        template <>
        struct ScalarTraits<BetaT> : public ScalarTraits<double>
        {
            using Base = ScalarTraits<double>;

            static void output(const BetaT& value, void* v, llvm::raw_ostream& out)
            {
                Base::output(value, v, out);
            }

            static StringRef input(StringRef scalar, void* v, BetaT& value)
            {
                return Base::input(scalar, v, (double&)(value));
            }
        };

        ///
        // Mapping of the test param elements of ContractionTestParams for reading / writing.
        ///
        template <>
        struct MappingTraits<hiptensor::ContractionTestParams>
        {
            static void mapping(IO& io, hiptensor::ContractionTestParams& doc)
            {
                // Logging bitfield
                io.mapRequired("Log Level", doc.logLevelMask());

                // Sequences of combinatorial fields
                io.mapRequired("Tensor Data Types", doc.dataTypes());
                io.mapRequired("Algorithm Types", doc.algorithms());
                io.mapRequired("Operators", doc.operators());
                io.mapRequired("Worksize Prefs", doc.workSizePrefrences());
                io.mapOptional("Alphas", (std::vector<std::vector<double>>&)(doc.alphas()));
                io.mapOptional("Betas",
                               (std::vector<std::vector<double>>&)(doc.betas()),
                               std::vector<std::vector<double>>(doc.alphas().size()));
                io.mapRequired(
                    "Lengths",
                    (std::vector<std::vector<std::vector<size_t>>>&)doc.problemLengths());
                io.mapRequired("Modes",
                               (std::vector<std::vector<std::vector<int32_t>>>&)doc.problemModes());

                // Default values for optional values
                auto defaultStrides
                    = std::vector<std::vector<std::vector<std::size_t>>>(doc.problemLengths());
                for(auto i = 0; i < defaultStrides.size(); i++)
                {
                    for(auto j = 0; j < defaultStrides[i].size(); j++)
                    {
                        defaultStrides[i][j] = std::vector<std::size_t>(
                            doc.problemLengths()[i][j].size(), std::size_t(0));
                    }
                }

                io.mapOptional("Strides", doc.problemStrides(), defaultStrides);
            }

            // Additional validation for input / output of the config
            static std::string validate(IO& io, hiptensor::ContractionTestParams& doc)
            {

                if(doc.problemLengths().size() == 0)
                {
                    return "Error: Empty Lengths";
                }

                if(doc.problemModes().size() == 0)
                {
                    return "Error: Empty Modes";
                }

                if(doc.alphas().size() == 0)
                {
                    return "Error: Empty Alphas";
                }

                if(std::any_of(doc.alphas().cbegin(), doc.alphas().cend(), [](auto&& alpha) {
                       return alpha.size() > 2 || alpha.size() <= 0;
                   }))
                {
                    return "Error: invalid Alpha";
                }

                if(doc.betas().size() > 0 && doc.betas().size() != doc.alphas().size())
                {
                    return "Error: Alphas and betas must have same size";
                }

                if(doc.problemStrides().size() > 1
                   && doc.problemStrides()[0].size() != doc.problemLengths()[0].size())
                {
                    return "Error: Problem strides and lengths must have same size";
                }

                if(doc.problemModes()[0].size() != doc.problemLengths()[0].size())
                {
                    return "Error: Problem modes and lengths must have same size";
                }

                return std::string{};
            }
        };

        ///
        // Mapping of the test param elements of PermutationTestParams for reading / writing.
        ///
        template <>
        struct MappingTraits<hiptensor::PermutationTestParams>
        {
            static void mapping(IO& io, hiptensor::PermutationTestParams& doc)
            {
                // Logging bitfield
                io.mapRequired("Log Level", doc.logLevelMask());

                // Sequences of combinatorial fields
                io.mapRequired("Tensor Data Types", doc.dataTypes());
                io.mapRequired("Alphas", (std::vector<AlphaT>&)(doc.alphas()));
                io.mapRequired("Lengths", doc.problemLengths());
                io.mapRequired("Permuted Dims", doc.permutedDims());
                io.mapRequired("Operators", (doc.operators()));
            }

            // Additional validation for input / output of the config
            static std::string validate(IO& io, hiptensor::PermutationTestParams& doc)
            {

                if(doc.problemLengths().size() == 0)
                {
                    return "Error: Empty Lengths";
                }

                if(doc.alphas().size() == 0)
                {
                    return "Error: Empty Alphas";
                }

                if(doc.permutedDims().size() == 0)
                {
                    return "Error: Empty Permuted Dims";
                }

                if(doc.operators().size() == 0)
                {
                    return "Error: Empty Operators";
                }

                return std::string{};
            }
        };

        ///
        // Mapping of the test param elements of ReductionTestParams for reading / writing.
        ///
        template <>
        struct MappingTraits<hiptensor::ReductionTestParams>
        {
            static void mapping(IO& io, hiptensor::ReductionTestParams& doc)
            {
                // Logging bitfield
                io.mapRequired("Log Level", doc.logLevelMask());

                // Sequences of combinatorial fields
                io.mapRequired("Tensor Data Types", doc.dataTypes());
                io.mapRequired("Alphas", (std::vector<AlphaT>&)(doc.alphas()));
                io.mapRequired("Betas", (std::vector<BetaT>&)(doc.betas()));
                io.mapRequired("Lengths", doc.problemLengths());
                io.mapRequired("Output Dims", doc.outputDims());
                io.mapRequired("Operators", (doc.operators()));
            }

            // Additional validation for input / output of the config
            static std::string validate(IO& io, hiptensor::ReductionTestParams& doc)
            {

                if(doc.problemLengths().size() == 0)
                {
                    return "Error: Empty Lengths";
                }

                if(doc.alphas().size() == 0)
                {
                    return "Error: Empty Alphas";
                }

                if(doc.betas().size() == 0)
                {
                    return "Error: Empty Betas";
                }

                if(doc.outputDims().size() == 0)
                {
                    return "Error: Empty Output Dims";
                }

                if(doc.operators().size() == 0)
                {
                    return "Error: Empty Operators";
                }

                return std::string{};
            }
        };

    } // namespace yaml

} // namespace llvm

// Instantiate the yaml loader for the ContractionTestParams and PermutationTestParams
namespace hiptensor
{
    template struct YamlConfigLoader<ContractionTestParams>;
    template struct YamlConfigLoader<PermutationTestParams>;
    template struct YamlConfigLoader<ReductionTestParams>;
}
