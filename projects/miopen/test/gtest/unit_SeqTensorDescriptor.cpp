/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <gtest/gtest.h>
#include <miopen/logger.hpp>
#include <miopen/seq_tensor.hpp>

namespace miopen {
namespace unit_tests {

struct SeqTensorDescriptorParams
{
    SeqTensorDescriptorParams(miopenDataType_t datatype_in, std::vector<size_t>&& lens_in)
        : datatype(datatype_in), lens(std::move(lens_in))
    {
    }

    SeqTensorDescriptorParams(miopenDataType_t datatype_in,
                              std::vector<size_t>&& lens_in,
                              bool with_padded_seq_layout)
        : datatype(datatype_in), lens(std::move(lens_in)), padded_seq_layout(with_padded_seq_layout)
    {
    }

    size_t GetNumDims() const { return lens.size(); }

    const std::vector<size_t>& GetLens() const { return lens; }

    miopenDataType_t GetDataType() const { return datatype; }

    SeqTensorDescriptor GetSeqTensorDescriptor() const
    {
        std::vector<unsigned int> layout_default(lens.size());
        std::iota(layout_default.begin(), layout_default.end(), 0);
        return {datatype, layout_default, lens, padded_seq_layout};
    }

    friend std::ostream& operator<<(std::ostream& os, const SeqTensorDescriptorParams& tp)
    {
        os << tp.datatype << ", ";
        miopen::LogRange(os << "{", tp.lens, ",") << "}, ";
        os << tp.padded_seq_layout;
        return os;
    }

private:
    miopenDataType_t datatype;
    std::vector<size_t> lens;
    bool padded_seq_layout;
};

} // namespace unit_tests
} // namespace miopen

struct TestCaseGetMaxCountOfSequence
{
    miopen::unit_tests::SeqTensorDescriptorParams tp;
    size_t actual_count_of_sequence;

    friend std::ostream& operator<<(std::ostream& os, const TestCaseGetMaxCountOfSequence& tc)
    {
        os << "(";
        os << "(" << tc.tp << "), ";
        os << tc.actual_count_of_sequence;
        os << ")";
        return os;
    }
};

struct TestCaseGetMaxSequenceLength
{
    miopen::unit_tests::SeqTensorDescriptorParams tp;
    size_t actual_sequence_length;

    friend std::ostream& operator<<(std::ostream& os, const TestCaseGetMaxSequenceLength& tc)
    {
        os << "(";
        os << "(" << tc.tp << "), ";
        os << tc.actual_sequence_length;
        os << ")";
        return os;
    }
};

struct TestCaseGetTotalSequenceLength
{
    miopen::unit_tests::SeqTensorDescriptorParams tp;
    size_t actual_sequence_length;

    friend std::ostream& operator<<(std::ostream& os, const TestCaseGetTotalSequenceLength& tc)
    {
        os << "(";
        os << "(" << tc.tp << "), ";
        os << tc.actual_sequence_length;
        os << ")";
        return os;
    }
};

struct TestGetMaxCountOfSequence : public ::testing::TestWithParam<TestCaseGetMaxCountOfSequence>
{
    static auto GetTestCases()
    {
        using TestCase = TestCaseGetMaxCountOfSequence;

        return std::vector{
            // clang-format off
            TestCase{{miopenHalf, {2, 2, 2}}, 2},
            TestCase{{miopenHalf, {2, 2, 2}, true}, 2},
            TestCase{{miopenHalf, {2, 2, 2, 2}}, 2},
            TestCase{{miopenHalf, {2, 2, 2, 2}, true}, 2},

            TestCase{{miopenHalf, {2, 8, 2, 2}}, 2},
            TestCase{{miopenHalf, {2, 8, 2, 2}, true}, 2},
            TestCase{{miopenHalf, {2, 16, 8, 2, 2}}, 2},
            TestCase{{miopenHalf, {2, 16, 8, 2, 2}, true}, 2},
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p  = GetParam();
        const auto td = p.tp.GetSeqTensorDescriptor();
        ASSERT_EQ(td.GetMaxCountOfSequences(), p.actual_count_of_sequence);
    }
};

struct TestGetMaxSequenceLength : public ::testing::TestWithParam<TestCaseGetMaxSequenceLength>
{
    static auto GetTestCases()
    {
        using TestCase = TestCaseGetMaxSequenceLength;

        return std::vector{
            // clang-format off
            TestCase{{miopenHalf, {2, 2, 2}}, 2},
            TestCase{{miopenHalf, {2, 2, 2}, true}, 2},
            TestCase{{miopenHalf, {2, 2, 2, 2}}, 2},
            TestCase{{miopenHalf, {2, 2, 2, 2}, true}, 2},

            TestCase{{miopenHalf, {2, 8, 2, 2}}, 8},
            TestCase{{miopenHalf, {2, 8, 2, 2}, true}, 8},
            TestCase{{miopenHalf, {2, 16, 8, 2, 2}}, 16},
            TestCase{{miopenHalf, {2, 16, 8, 2, 2}, true}, 16},
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p  = GetParam();
        const auto td = p.tp.GetSeqTensorDescriptor();
        ASSERT_EQ(td.GetMaxSequenceLength(), p.actual_sequence_length);
    }
};

struct TestGetTotalSequenceLength : public ::testing::TestWithParam<TestCaseGetTotalSequenceLength>
{
    static auto GetTestCases()
    {
        using TestCase = TestCaseGetTotalSequenceLength;

        return std::vector{
            // clang-format off
            TestCase{{miopenHalf, {2, 2, 2}}, 4},
            TestCase{{miopenHalf, {2, 2, 2}, true}, 4},
            TestCase{{miopenHalf, {2, 2, 2, 2}}, 4},
            TestCase{{miopenHalf, {2, 2, 2, 2}, true}, 4},

            TestCase{{miopenHalf, {2, 8, 2, 2}}, 16},
            TestCase{{miopenHalf, {2, 8, 2, 2}, true}, 16},
            TestCase{{miopenHalf, {2, 16, 8, 2, 2}}, 32},
            TestCase{{miopenHalf, {2, 16, 8, 2, 2}, true}, 32},
            // clang-format on
        };
    }

    void RunTest()
    {
        const auto p  = GetParam();
        const auto td = p.tp.GetSeqTensorDescriptor();
        ASSERT_EQ(td.GetTotalSequenceLen(), p.actual_sequence_length);
    }
};

using CPU_TestGetMaxCountOfSequence_FP16  = TestGetMaxCountOfSequence;
using CPU_TestGetMaxSequenceLength_FP16   = TestGetMaxSequenceLength;
using CPU_TestGetTotalSequenceLength_FP16 = TestGetTotalSequenceLength;

TEST_P(CPU_TestGetMaxCountOfSequence_FP16, SeqTensorDescriptor) { this->RunTest(); };
TEST_P(CPU_TestGetMaxSequenceLength_FP16, SeqTensorDescriptor) { this->RunTest(); };
TEST_P(CPU_TestGetTotalSequenceLength_FP16, SeqTensorDescriptor) { this->RunTest(); };

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TestGetMaxCountOfSequence_FP16,
                         testing::ValuesIn(TestGetMaxCountOfSequence::GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TestGetMaxSequenceLength_FP16,
                         testing::ValuesIn(TestGetMaxSequenceLength::GetTestCases()));

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_TestGetTotalSequenceLength_FP16,
                         testing::ValuesIn(TestGetTotalSequenceLength::GetTestCases()));
