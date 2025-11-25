/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
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


#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include <DataGenerator.hpp>

using std::vector;
using ::testing::TestWithParam;

using namespace DGen;

using DataGeneratorTypes = ::testing::Types<f32,
                                            fp16,
                                            bf16,
                                            ocp_e4m3_mxfp8,
                                            ocp_e5m2_mxfp8,
                                            ocp_e2m3_mxfp6,
                                            ocp_e3m2_mxfp6,
                                            ocp_e2m1_mxfp4>;

typedef std::tuple<bool, bool, bool, bool, vector<double>, DataScaling, vector<index_t>>
    BoundedTupleType;
typedef std::tuple<bool, bool, bool, bool, double, DataScaling, vector<index_t>>
    BoundedAlternatingSignTupleType;
typedef std::tuple<bool, bool, bool, bool, DataScaling, vector<index_t>> UnboundedTupleType;
typedef UnboundedTupleType                                           TrigonometricTupleType;
typedef std::tuple<bool, DataScaling, vector<index_t>>                   ZerosTupleType;
typedef ZerosTupleType                                               OnesTupleType;
typedef ZerosTupleType                                               IdentityTupleType;

// clampToF32
const vector<bool> clamp_params = {false, true};

// includeInf
const vector<bool> inf_params = {false, true};

// includeNaN
const vector<bool> nan_params = {false, true};

// forceDenorm
const vector<bool> denorm_params = {false, true};

// DataScaling
const vector<DataScaling> scale_params = {Mean};

// block size, size, stride
const vector<vector<index_t>> dim_params = {
    {5, 16, 10, 10, 1},
    {5, 10, 16, 1, 10},
    {4, 4, 4, 1, 8},
    // {256, 1024, 1024, 1, 1024},
    {16, 64, 64, 1, 64},
    {3, 3, 3, 3, 1, 3, 9},
    {1, 2, 1, 2, 2, 1, 1},
    {2, 10, 2, 2, 4, 2, 1},
};

const vector<vector<index_t>> two_dim_params = {
    {5, 16, 10, 10, 1},
    {5, 10, 16, 1, 10},
    {4, 4, 4, 1, 8},
    // {256, 1024, 1024, 1, 1024},
    {16, 64, 64, 1, 64},
};

// min/max
const vector<vector<double>> min_max_params = {
    {-1.0, 1.0},
    {0.0, 1.0},
    {-1.0, 0.0},
    {std::numeric_limits<float>::min(), std::numeric_limits<float>::max()},
    {std::numeric_limits<double>::min(), std::numeric_limits<double>::max()},
    {-std::numeric_limits<float>::max(), std::numeric_limits<float>::max()},
    {-std::numeric_limits<double>::max(), std::numeric_limits<double>::max()},
};

const vector<vector<double>> min_max_denorm_params = {
    {-getDataMaxSubnorm<f32>(), getDataMaxSubnorm<f32>()},
    {-getDataMaxSubnorm<fp16>(), getDataMaxSubnorm<fp16>()},
    {-getDataMaxSubnorm<bf16>(), getDataMaxSubnorm<bf16>()},
    {-getDataMaxSubnorm<ocp_e2m1_mxfp4>(), getDataMaxSubnorm<ocp_e2m1_mxfp4>()},
    {-getDataMaxSubnorm<ocp_e2m3_mxfp6>(), getDataMaxSubnorm<ocp_e2m3_mxfp6>()},
    {-getDataMaxSubnorm<ocp_e3m2_mxfp6>(), getDataMaxSubnorm<ocp_e3m2_mxfp6>()},
    {-getDataMaxSubnorm<ocp_e4m3_mxfp8>(), getDataMaxSubnorm<ocp_e4m3_mxfp8>()},
    {-getDataMaxSubnorm<ocp_e5m2_mxfp8>(), getDataMaxSubnorm<ocp_e5m2_mxfp8>()},
};

// max
const vector<double> max_params
    = {1.0, -1.0, std::numeric_limits<float>::max(), std::numeric_limits<double>::max()};

const vector<double> max_denorm_params = {getDataMaxSubnorm<f32>(),
                                          getDataMaxSubnorm<fp16>(),
                                          getDataMaxSubnorm<bf16>(),
                                          getDataMaxSubnorm<ocp_e2m1_mxfp4>(),
                                          getDataMaxSubnorm<ocp_e2m3_mxfp6>(),
                                          getDataMaxSubnorm<ocp_e3m2_mxfp6>(),
                                          getDataMaxSubnorm<ocp_e4m3_mxfp8>(),
                                          getDataMaxSubnorm<ocp_e5m2_mxfp8>()};

void set_block_size_stride(const vector<index_t>& dims,
                           index_t&               block_scale,
                           vector<index_t>&       size,
                           vector<index_t>&       stride)
{
    assert(dims.size() % 2 == 1);

    block_scale = dims[0];

    const auto n = (dims.size() - 1) / 2 + 1;
    size         = vector<index_t>(dims.begin() + 1, dims.begin() + n);
    stride       = vector<index_t>(dims.begin() + n, dims.end());
}

std::ostream& operator<<(std::ostream& os, const DataGeneratorOptions& opts)
{
    os << std::boolalpha;

    os << "clampToF32{" << opts.clampToF32 << "} ";
    os << "includeInf{" << opts.includeInf << "} ";
    os << "includeNaN{" << opts.includeNaN << "} ";
    os << "forceDenorm{" << opts.forceDenorm << "} ";
    os << std::noboolalpha;

    os << "(min, max)=(" << opts.min << ", ";
    os << opts.max << ") ";

    os << "blockScaling{" << opts.blockScaling << "} ";
    os << "scaling{" << opts.scaling << "} ";

    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<index_t>& vec)
{

    os << "{ ";
    for(const auto v : vec)
    {
        os << v << ", ";
    }
    os << "}";
    return os;
}

template <typename DataType>
class DataGeneratorBoundedTest : public ::TestWithParam<BoundedTupleType>
{
    void set_options(BoundedTupleType      tup,
                     DataGeneratorOptions& opts,
                     vector<index_t>&          size,
                     vector<index_t>&          stride)
    {
        opts.clampToF32  = std::get<0>(tup);
        opts.includeInf  = std::get<1>(tup);
        opts.includeNaN  = std::get<2>(tup);
        opts.forceDenorm = std::get<3>(tup);

        opts.min = std::get<4>(tup)[0];
        opts.max = std::get<4>(tup)[1];

        opts.scaling = std::get<5>(tup);

        set_block_size_stride(std::get<6>(tup), opts.blockScaling, size, stride);
    }

public:
    void testForDataType(BoundedTupleType& params)
    {
        DataGeneratorOptions opts;
        vector<index_t>          size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.pattern = Bounded;

        const auto dgen  = DataGenerator<DataType>().generate(size, stride, opts);
        const auto data  = dgen.getDataBytes();
        const auto scale = dgen.getScaleBytes();

        const auto ref_double = dgen.getReferenceDouble();
        const auto ref_float  = dgen.getReferenceFloat();

        auto total_size = size[0];
        for(index_t i = 1; i < size.size(); i++)
        {
            total_size *= size[i];
        }

        const int              num_threads_test
            = (std::thread::hardware_concurrency() > 32) ? 32 : std::thread::hardware_concurrency();

        vector<bool> has_nan(num_threads_test, false);
        vector<bool> has_inf(num_threads_test, false);
        vector<bool> has_sbn(num_threads_test, false);

        // check values
        #pragma omp parallel for num_threads(num_threads_test)
        for(index_t i = 0; i < total_size; i++)
        {
            // find position
            index_t data_i = (i % size[size.size() - 1]) * stride[size.size() - 1];

            auto tmp = i / size[size.size() - 1];
            for(index_t j = size.size() - 2; j > 0; j--)
            {
                data_i += (tmp % size[j]) * stride[j];
                tmp /= size[j];
            }

            data_i += tmp * stride[0];

            const index_t scale_i = data_i / opts.blockScaling;

            // test
            const auto ref_value = toDoublePacked<DataType>(&scale[0], &data[0], scale_i, data_i);
            const auto abs_ref_value = std::abs(ref_value);

            if(!std::isnan(ref_value) && !std::isinf(ref_value))
            {
                EXPECT_GE(ref_value, opts.min);
                EXPECT_LE(ref_value, opts.max);

                if(opts.clampToF32 && ref_value != 0)
                {
                    EXPECT_GE(abs_ref_value, std::numeric_limits<float>::denorm_min());
                    EXPECT_LE(abs_ref_value, std::numeric_limits<float>::max());
                }
            }

            EXPECT_TRUE(opts.includeNaN || !std::isnan(ref_value));
            EXPECT_TRUE(opts.includeInf || !std::isinf(ref_value));

            // test reference
            if(!std::isnan(ref_value))
            {
                EXPECT_EQ(ref_double[data_i], ref_value);
                EXPECT_EQ(ref_float[data_i],
                          toFloatPacked<DataType>(&scale[0], &data[0], scale_i, data_i));
            }
            else
            {
                EXPECT_TRUE(std::isnan(ref_double[data_i]));
                EXPECT_TRUE(std::isnan(ref_float[data_i]));
            }

            const auto tid            = omp_get_thread_num();
            if (isNaNPacked<DataType>(&scale[0], &data[0], scale_i, data_i))
                has_nan[tid] = true;
            if (isInfPacked<DataType>(&scale[0], &data[0], scale_i, data_i))
                has_inf[tid] = true;
            if (isSubnormPacked<DataType>(&data[0], data_i))
                has_sbn[tid] = true;
        }

        if(opts.includeNaN && DataType::dataInfo.hasNan)
        {
            ASSERT_TRUE(std::any_of(has_nan.begin(), has_nan.end(), [](bool v){ return v;}));
        }

        if(opts.includeInf && DataType::dataInfo.hasInf)
        {
            ASSERT_TRUE(std::any_of(has_inf.begin(), has_inf.end(), [](bool v){ return v;}));
        }

        if(opts.forceDenorm && isScaled<DataType>()
           && ((opts.min < getDataMinSubnorm<DataType>()
                && opts.max > getDataMinSubnorm<DataType>())
               || (opts.min < -getDataMinSubnorm<DataType>()
                   && opts.max > -getDataMinSubnorm<DataType>())))
        {
            ASSERT_TRUE(std::any_of(has_sbn.begin(), has_sbn.end(), [](bool v){ return v;}));
        }
    }
};

template <typename DataType>
class DataGeneratorBoundedAlternatingSignTest
    : public ::TestWithParam<BoundedAlternatingSignTupleType>
{
    void set_options(BoundedAlternatingSignTupleType tup,
                     DataGeneratorOptions&           opts,
                     vector<index_t>&                    size,
                     vector<index_t>&                    stride)
    {
        opts.clampToF32  = std::get<0>(tup);
        opts.includeInf  = std::get<1>(tup);
        opts.includeNaN  = std::get<2>(tup);
        opts.forceDenorm = std::get<3>(tup);

        opts.max = std::get<4>(tup);

        opts.scaling = std::get<5>(tup);

        set_block_size_stride(std::get<6>(tup), opts.blockScaling, size, stride);
    }

public:
    void testForDataType(BoundedAlternatingSignTupleType& params)
    {
        DataGeneratorOptions opts;
        vector<index_t>          size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.pattern = BoundedAlternatingSign;

        const auto dgen  = DataGenerator<DataType>().generate(size, stride, opts);
        const auto data  = dgen.getDataBytes();
        const auto scale = dgen.getScaleBytes();

        const auto ref_double = dgen.getReferenceDouble();
        const auto ref_float  = dgen.getReferenceFloat();

        auto total_size = size[0];
        for(index_t i = 1; i < size.size(); i++)
        {
            total_size *= size[i];
        }

        bool has_nan = false;
        bool has_inf = false;
        bool has_sbn = false;

        // check values
        for(index_t i = 0; i < total_size; i++)
        {
            // find position
            index_t data_i = (i % size[size.size() - 1]) * stride[size.size() - 1];

            auto tmp = i / size[size.size() - 1];
            for(index_t j = size.size() - 2; j > 0; j--)
            {
                data_i += (tmp % size[j]) * stride[j];
                tmp /= size[j];
            }

            data_i += tmp * stride[0];

            const index_t scale_i = data_i / opts.blockScaling;

            // test
            const auto ref_value = toDoublePacked<DataType>(&scale[0], &data[0], scale_i, data_i);
            const auto abs_ref_value = std::abs(ref_value);

            if(!std::isnan(ref_value) && !std::isinf(ref_value))
            {
                EXPECT_LE(abs_ref_value, std::abs(opts.max));

                if(ref_value != 0)
                {
                    if(data_i % 2)
                    {
                        EXPECT_TRUE(std::signbit(ref_value));
                    }
                    else
                    {
                        EXPECT_FALSE(std::signbit(ref_value));
                    }
                }

                if(opts.clampToF32 && ref_value != 0)
                {
                    EXPECT_GE(abs_ref_value, std::numeric_limits<float>::denorm_min());
                    EXPECT_LE(abs_ref_value, std::numeric_limits<float>::max());
                }
            }

            EXPECT_TRUE(opts.includeNaN || !std::isnan(ref_value));
            EXPECT_TRUE(opts.includeInf || !std::isinf(ref_value));

            // test reference
            if(!std::isnan(ref_value))
            {
                EXPECT_EQ(ref_double[data_i], ref_value);
                EXPECT_EQ(ref_float[data_i],
                          toFloatPacked<DataType>(&scale[0], &data[0], scale_i, data_i));
            }
            else
            {
                EXPECT_TRUE(std::isnan(ref_double[data_i]));
                EXPECT_TRUE(std::isnan(ref_float[data_i]));
            }

            has_nan = has_nan || isNaNPacked<DataType>(&scale[0], &data[0], scale_i, data_i);
            has_inf = has_inf || isInfPacked<DataType>(&scale[0], &data[0], scale_i, data_i);
            has_sbn = has_sbn || isSubnormPacked<DataType>(&data[0], data_i);
        }

        if(opts.includeNaN && getDataHasNan<DataType>())
        {
            EXPECT_TRUE(has_nan);
        }

        if(opts.includeInf && getDataHasInf<DataType>())
        {
            EXPECT_TRUE(has_inf);
        }

        if(opts.forceDenorm && isScaled<DataType>()
           && (opts.max > getDataMinSubnorm<DataType>()
               || opts.max > -getDataMinSubnorm<DataType>()))
        {
            EXPECT_TRUE(has_sbn);
        }
    }
};

template <typename DataType>
class DataGeneratorUnboundedTest : public ::TestWithParam<UnboundedTupleType>
{
    void set_options(UnboundedTupleType    tup,
                     DataGeneratorOptions& opts,
                     vector<index_t>&          size,
                     vector<index_t>&          stride)
    {
        opts.clampToF32  = std::get<0>(tup);
        opts.includeInf  = std::get<1>(tup);
        opts.includeNaN  = std::get<2>(tup);
        opts.forceDenorm = std::get<3>(tup);

        opts.scaling = std::get<4>(tup);

        set_block_size_stride(std::get<5>(tup), opts.blockScaling, size, stride);
    }

public:
    void testForDataType(UnboundedTupleType& params)
    {
        DataGeneratorOptions opts;
        vector<index_t>          size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.pattern = Unbounded;

        const auto dgen  = DataGenerator<DataType>().generate(size, stride, opts);
        const auto data  = dgen.getDataBytes();
        const auto scale = dgen.getScaleBytes();

        const auto ref_double = dgen.getReferenceDouble();
        const auto ref_float  = dgen.getReferenceFloat();

        index_t total_size = size[0];
        for(index_t i = 1; i < size.size(); i++)
        {
            total_size *= size[i];
        }

        bool has_nan = false;
        bool has_inf = false;
        bool has_sbn = false;

        // check values
        for(index_t i = 0; i < total_size; i++)
        {
            // find position
            index_t data_i = (i % size[size.size() - 1]) * stride[size.size() - 1];

            auto tmp = i / size[size.size() - 1];
            for(index_t j = size.size() - 2; j > 0; j--)
            {
                data_i += (tmp % size[j]) * stride[j];
                tmp /= size[j];
            }

            data_i += tmp * stride[0];

            const index_t scale_i = data_i / opts.blockScaling;

            // test
            const auto ref_value = toDoublePacked<DataType>(&scale[0], &data[0], scale_i, data_i);
            const auto abs_ref_value = std::abs(ref_value);

            if(opts.clampToF32 && ref_value != 0 && !std::isnan(ref_value)
               && !std::isinf(ref_value))
            {
                EXPECT_GE(abs_ref_value, std::numeric_limits<float>::denorm_min());
                EXPECT_LE(abs_ref_value, std::numeric_limits<float>::max());
            }

            EXPECT_TRUE(opts.includeNaN || !std::isnan(ref_value));
            EXPECT_TRUE(opts.includeInf || !std::isinf(ref_value));

            // test reference
            if(!std::isnan(ref_value))
            {
                EXPECT_EQ(ref_double[data_i], ref_value);
                EXPECT_EQ(ref_float[data_i],
                          toFloatPacked<DataType>(&scale[0], &data[0], scale_i, data_i));
            }
            else
            {
                EXPECT_TRUE(std::isnan(ref_double[data_i]));
                EXPECT_TRUE(std::isnan(ref_float[data_i]));
            }

            has_nan = has_nan || isNaNPacked<DataType>(&scale[0], &data[0], scale_i, data_i);
            has_inf = has_inf || isInfPacked<DataType>(&scale[0], &data[0], scale_i, data_i);
            has_sbn = has_sbn || isSubnormPacked<DataType>(&data[0], data_i);
        }

        if(opts.includeNaN && getDataHasNan<DataType>())
        {
            EXPECT_TRUE(has_nan);
        }

        if(opts.includeInf && getDataHasInf<DataType>())
        {
            EXPECT_TRUE(has_inf);
        }

        if(opts.forceDenorm && isScaled<DataType>())
        {
            EXPECT_TRUE(has_sbn);
        }
    }
};

template <typename DataType>
class DataGeneratorTrigonometricTest : public ::TestWithParam<TrigonometricTupleType>
{
    void set_options(TrigonometricTupleType tup,
                     DataGeneratorOptions&  opts,
                     vector<index_t>&           size,
                     vector<index_t>&           stride)
    {
        opts.clampToF32  = std::get<0>(tup);
        opts.includeInf  = std::get<1>(tup);
        opts.includeNaN  = std::get<2>(tup);
        opts.forceDenorm = std::get<3>(tup);

        opts.scaling = std::get<4>(tup);

        set_block_size_stride(std::get<5>(tup), opts.blockScaling, size, stride);
    }

public:
    void testForDataType(TrigonometricTupleType& params)
    {
        DataGeneratorOptions opts;
        vector<index_t>          size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.pattern = Trigonometric;

        const auto dgen  = DataGenerator<DataType>().generate(size, stride, opts);
        const auto data  = dgen.getDataBytes();
        const auto scale = dgen.getScaleBytes();

        const auto ref_double = dgen.getReferenceDouble();
        const auto ref_float  = dgen.getReferenceFloat();

        index_t total_size = size[0];
        for(index_t i = 1; i < size.size(); i++)
        {
            total_size *= size[i];
        }

        bool has_nan = false;
        bool has_inf = false;
        bool has_sbn = false;

        // check values
        for(index_t i = 0; i < total_size; i++)
        {
            // find position
            index_t data_i = (i % size[size.size() - 1]) * stride[size.size() - 1];

            auto tmp = i / size[size.size() - 1];
            for(index_t j = size.size() - 2; j > 0; j--)
            {
                data_i += (tmp % size[j]) * stride[j];
                tmp /= size[j];
            }

            data_i += tmp * stride[0];

            const index_t scale_i = data_i / opts.blockScaling;

            // test
            const auto ref_value = toDoublePacked<DataType>(&scale[0], &data[0], scale_i, data_i);
            const auto abs_ref_value = std::abs(ref_value);

            if(!std::isnan(ref_value) && !std::isinf(ref_value))
            {
                EXPECT_GE(ref_value, -1.0);
                EXPECT_LE(ref_value, 1.0);

                if(opts.clampToF32 && ref_value != 0)
                {
                    EXPECT_GE(abs_ref_value, std::numeric_limits<float>::denorm_min());
                    EXPECT_LE(abs_ref_value, std::numeric_limits<float>::max());
                }
            }

            EXPECT_TRUE(opts.includeNaN || !std::isnan(ref_value));
            EXPECT_TRUE(opts.includeInf || !std::isinf(ref_value));

            // test reference
            if(!std::isnan(ref_value))
            {
                EXPECT_EQ(ref_double[data_i], ref_value);
                EXPECT_EQ(ref_float[data_i],
                          toFloatPacked<DataType>(&scale[0], &data[0], scale_i, data_i));
            }
            else
            {
                EXPECT_TRUE(std::isnan(ref_double[data_i]));
                EXPECT_TRUE(std::isnan(ref_float[data_i]));
            }

            has_nan = has_nan || isNaNPacked<DataType>(&scale[0], &data[0], scale_i, data_i);
            has_inf = has_inf || isInfPacked<DataType>(&scale[0], &data[0], scale_i, data_i);
            has_sbn = has_sbn || isSubnormPacked<DataType>(&data[0], data_i);
        }

        if(opts.includeNaN && getDataHasNan<DataType>())
        {
            EXPECT_TRUE(has_nan);
        }

        if(opts.includeInf && getDataHasInf<DataType>())
        {
            EXPECT_TRUE(has_inf);
        }

        if(opts.forceDenorm && isScaled<DataType>())
        {
            EXPECT_TRUE(has_sbn);
        }
    }
};

template <typename DataType>
class DataGeneratorZerosTest : public ::TestWithParam<ZerosTupleType>
{
    void set_options(ZerosTupleType        tup,
                     DataGeneratorOptions& opts,
                     vector<index_t>&          size,
                     vector<index_t>&          stride)
    {
        opts.forceDenorm = std::get<0>(tup);
        opts.scaling     = std::get<1>(tup);

        set_block_size_stride(std::get<2>(tup), opts.blockScaling, size, stride);
    }

public:
    void testForDataType(ZerosTupleType& params)
    {
        DataGeneratorOptions opts;
        vector<index_t>          size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.pattern = Zeros;

        const auto dgen  = DataGenerator<DataType>().generate(size, stride, opts);
        const auto data  = dgen.getDataBytes();
        const auto scale = dgen.getScaleBytes();

        const auto ref_double = dgen.getReferenceDouble();
        const auto ref_float  = dgen.getReferenceFloat();

        index_t total_size = size[0];
        for(index_t i = 1; i < size.size(); i++)
        {
            total_size *= size[i];
        }

        // check values
        for(index_t i = 0; i < total_size; i++)
        {
            // find position
            index_t data_i = (i % size[size.size() - 1]) * stride[size.size() - 1];

            auto tmp = i / size[size.size() - 1];
            for(index_t j = size.size() - 2; j > 0; j--)
            {
                data_i += (tmp % size[j]) * stride[j];
                tmp /= size[j];
            }

            data_i += tmp * stride[0];

            const index_t scale_i = data_i / opts.blockScaling;

            // test
            EXPECT_TRUE(isZeroPacked<DataType>(&scale[0], &data[0], scale_i, data_i));

            // test reference
            EXPECT_EQ(ref_double[data_i],
                      toDoublePacked<DataType>(&scale[0], &data[0], scale_i, data_i));
            EXPECT_EQ(ref_float[data_i],
                      toFloatPacked<DataType>(&scale[0], &data[0], scale_i, data_i));
        }
    }
};

template <typename DataType>
class DataGeneratorOnesTest : public ::TestWithParam<OnesTupleType>
{
    void set_options(OnesTupleType         tup,
                     DataGeneratorOptions& opts,
                     vector<index_t>&          size,
                     vector<index_t>&          stride)
    {
        opts.forceDenorm = std::get<0>(tup);
        opts.scaling     = std::get<1>(tup);

        set_block_size_stride(std::get<2>(tup), opts.blockScaling, size, stride);
    }

public:
    void testForDataType(OnesTupleType& params)
    {
        DataGeneratorOptions opts;
        vector<index_t>          size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.pattern = Ones;

        const auto dgen  = DataGenerator<DataType>().generate(size, stride, opts);
        const auto data  = dgen.getDataBytes();
        const auto scale = dgen.getScaleBytes();

        const auto ref_double = dgen.getReferenceDouble();
        const auto ref_float  = dgen.getReferenceFloat();

        index_t total_size = size[0];
        for(index_t i = 1; i < size.size(); i++)
        {
            total_size *= size[i];
        }

        bool has_sbn = false;

        // check values
        for(index_t i = 0; i < total_size; i++)
        {
            // find position
            index_t data_i = (i % size[size.size() - 1]) * stride[size.size() - 1];

            auto tmp = i / size[size.size() - 1];
            for(index_t j = size.size() - 2; j > 0; j--)
            {
                data_i += (tmp % size[j]) * stride[j];
                tmp /= size[j];
            }

            data_i += tmp * stride[0];

            const index_t scale_i = data_i / opts.blockScaling;

            // test
            EXPECT_TRUE(isOnePacked<DataType>(&scale[0], &data[0], scale_i, data_i));

            // test reference
            EXPECT_EQ(ref_double[data_i],
                      toDoublePacked<DataType>(&scale[0], &data[0], scale_i, data_i));
            EXPECT_EQ(ref_float[data_i],
                      toFloatPacked<DataType>(&scale[0], &data[0], scale_i, data_i));

            has_sbn = has_sbn || isSubnormPacked<DataType>(&data[0], data_i);
        }

        if(opts.forceDenorm && isScaled<DataType>())
        {
            EXPECT_TRUE(has_sbn);
        }
    }
};

template <typename DataType>
class DataGeneratorIdentityTest : public ::TestWithParam<IdentityTupleType>
{
    void set_options(IdentityTupleType     tup,
                     DataGeneratorOptions& opts,
                     vector<index_t>&          size,
                     vector<index_t>&          stride)
    {
        opts.forceDenorm = std::get<0>(tup);
        opts.scaling     = std::get<1>(tup);

        set_block_size_stride(std::get<2>(tup), opts.blockScaling, size, stride);
    }

public:
    void testForDataType(IdentityTupleType& params)
    {
        DataGeneratorOptions opts;
        vector<index_t>          size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.pattern = Identity;

        const auto dgen  = DataGenerator<DataType>().generate(size, stride, opts);
        const auto data  = dgen.getDataBytes();
        const auto scale = dgen.getScaleBytes();

        const auto ref_double = dgen.getReferenceDouble();
        const auto ref_float  = dgen.getReferenceFloat();

        index_t total_size = size[0];
        for(index_t i = 1; i < size.size(); i++)
        {
            total_size *= size[i];
        }

        bool has_sbn = false;

        // check values
        for(index_t i = 0; i < total_size; i++)
        {
            // find position
            bool   diag     = true;
            index_t past_idx = i % size[size.size() - 1];

            index_t data_i = past_idx * stride[size.size() - 1];

            auto tmp = i / size[size.size() - 1];
            for(index_t j = size.size() - 2; j > 0; j--)
            {
                index_t curr_idx = (tmp % size[j]);
                diag            = diag && (past_idx == curr_idx);

                data_i += past_idx * stride[j];
                tmp /= size[j];

                past_idx = curr_idx;
            }

            data_i += tmp * stride[0];
            diag = diag && (past_idx == tmp);

            const index_t scale_i = data_i / opts.blockScaling;

            // test
            if(diag)
                EXPECT_TRUE(isOnePacked<DataType>(&scale[0], &data[0], scale_i, data_i));
            else
                EXPECT_TRUE(isZeroPacked<DataType>(&scale[0], &data[0], scale_i, data_i));

            // test reference
            EXPECT_EQ(ref_double[data_i],
                      toDoublePacked<DataType>(&scale[0], &data[0], scale_i, data_i));
            EXPECT_EQ(ref_float[data_i],
                      toFloatPacked<DataType>(&scale[0], &data[0], scale_i, data_i));

            has_sbn = has_sbn || isSubnormPacked<DataType>(&data[0], data_i);
        }

        if(opts.forceDenorm && isScaled<DataType>())
        {
            EXPECT_TRUE(has_sbn);
        }
    }
};

TYPED_TEST_SUITE(DataGeneratorBoundedTest, DataGeneratorTypes);
TYPED_TEST_SUITE(DataGeneratorBoundedAlternatingSignTest, DataGeneratorTypes);
TYPED_TEST_SUITE(DataGeneratorUnboundedTest, DataGeneratorTypes);
TYPED_TEST_SUITE(DataGeneratorTrigonometricTest, DataGeneratorTypes);
TYPED_TEST_SUITE(DataGeneratorZerosTest, DataGeneratorTypes);
TYPED_TEST_SUITE(DataGeneratorOnesTest, DataGeneratorTypes);
TYPED_TEST_SUITE(DataGeneratorIdentityTest, DataGeneratorTypes);

#define begin_end(container) begin(container), end(container)

TYPED_TEST(DataGeneratorBoundedTest, LargeBuffer)
{
    // This test tries to generate a large MxN matrix.
    index_t M = 65536;
    index_t N = 65536;

    BoundedTupleType params = {/*clamp*/ false,
                               /*inf*/ false,
                               /*nan*/ false,
                               /*denorm*/ false,
                               /*min/max*/ {-1.0, 1.0},
                               /*scale*/ {DataScaling::Mean},
                               /*dim*/ {{1, M, N, 1, M}}};

    this->testForDataType(params);
}

TYPED_TEST(DataGeneratorBoundedTest, TestForEachDataType)
{
    std::vector<BoundedTupleType> params;
    cartesian_product(params,
                      begin_end(clamp_params),
                      begin_end(inf_params),
                      begin_end(nan_params),
                      begin_end(denorm_params),
                      begin_end(min_max_params),
                      begin_end(scale_params),
                      begin_end(dim_params));
    for(auto v : params)
    {
        this->testForDataType(v);
    }
}

TYPED_TEST(DataGeneratorBoundedTest, TestForEachDataTypeDenormals)
{
    std::vector<BoundedTupleType> params;
    cartesian_product(params,
                      begin_end(clamp_params),
                      begin_end(inf_params),
                      begin_end(nan_params),
                      begin_end(denorm_params),
                      begin_end(min_max_denorm_params),
                      begin_end(scale_params),
                      begin_end(dim_params));
    for(auto v : params)
    {
        this->testForDataType(v);
    }
}

TYPED_TEST(DataGeneratorBoundedAlternatingSignTest, TestForEachDataType)
{
    std::vector<BoundedAlternatingSignTupleType> params;
    cartesian_product(params,
                      begin_end(clamp_params),
                      begin_end(inf_params),
                      begin_end(nan_params),
                      begin_end(denorm_params),
                      begin_end(max_params),
                      begin_end(scale_params),
                      begin_end(dim_params));
    for(auto v : params)
    {
        this->testForDataType(v);
    }
}

TYPED_TEST(DataGeneratorBoundedAlternatingSignTest, TestForEachDataTypeDenormals)
{
    std::vector<BoundedAlternatingSignTupleType> params;
    cartesian_product(params,
                      begin_end(clamp_params),
                      begin_end(inf_params),
                      begin_end(nan_params),
                      begin_end(denorm_params),
                      begin_end(max_denorm_params),
                      begin_end(scale_params),
                      begin_end(dim_params));
    for(auto v : params)
    {
        this->testForDataType(v);
    }
}

TYPED_TEST(DataGeneratorUnboundedTest, TestForEachDataType)
{
    std::vector<UnboundedTupleType> params;
    cartesian_product(params,
                      begin_end(clamp_params),
                      begin_end(inf_params),
                      begin_end(nan_params),
                      begin_end(denorm_params),
                      begin_end(scale_params),
                      begin_end(dim_params));
    for(auto v : params)
    {
        this->testForDataType(v);
    }
}

TYPED_TEST(DataGeneratorTrigonometricTest, TestForEachDataType)
{
    std::vector<TrigonometricTupleType> params;
    cartesian_product(params,
                      begin_end(clamp_params),
                      begin_end(inf_params),
                      begin_end(nan_params),
                      begin_end(denorm_params),
                      begin_end(scale_params),
                      begin_end(dim_params));
    for(auto v : params)
    {
        this->testForDataType(v);
    }
}

TYPED_TEST(DataGeneratorZerosTest, TestForEachDataType)
{
    std::vector<ZerosTupleType> params;
    cartesian_product(
        params, begin_end(denorm_params), begin_end(scale_params), begin_end(dim_params));
    for(auto v : params)
    {
        this->testForDataType(v);
    }
}

TYPED_TEST(DataGeneratorOnesTest, TestForEachDataType)
{
    std::vector<OnesTupleType> params;
    cartesian_product(
        params, begin_end(denorm_params), begin_end(scale_params), begin_end(dim_params));
    for(auto v : params)
    {
        this->testForDataType(v);
    }
}

TYPED_TEST(DataGeneratorIdentityTest, TestForEachDataType)
{
    std::vector<IdentityTupleType> params;
    cartesian_product(
        params, begin_end(denorm_params), begin_end(scale_params), begin_end(two_dim_params));
    for(auto v : params)
    {
        this->testForDataType(v);
    }
}
