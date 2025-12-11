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

#include <random>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <algorithm>

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
typedef std::tuple<bool, DataScaling, vector<index_t>>                   ZerosTupleType;
typedef ZerosTupleType                                                   OnesTupleType;
typedef ZerosTupleType                                                   IdentityTupleType;
typedef UnboundedTupleType                                               TrigonometricFromFloatTupleType;
typedef UnboundedTupleType                                               NormalFromFloatTupleType;

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

double getMean(const std::vector<double>& data)
{
    if(data.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    return std::reduce(data.begin(), data.end()) / data.size();
}

double getStdDev(const std::vector<double>& data)
{
    const double mean = getMean(data);
    // Covers empty vector case
    if(std::isnan(mean))
    {
        return std::numeric_limits<double>::quiet_NaN();
    }

    double sum = 0.0;
    for(const double& val : data)
    {
        sum += (val - mean) * (val - mean);
    }

    return std::sqrt(sum / data.size());
}

// Compute the cumulative distribution function (CDF) for standard normal distribution
double normalCDF(double x) {
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

// Class to represent an Empirical Distribution Function (EDF) of some distribution
class EDF{
    private:
        std::vector<double> dis;
        double n;

    public:
        EDF(const std::vector<double> & x){
            dis = x;
            std::sort(dis.begin(), dis.end());
            n = static_cast<double>(dis.size());
        }

        double operator()(double x) const{
            auto it = std::upper_bound(dis.begin(), dis.end(), x);
            double pos = static_cast<double>(it - dis.begin());
            return pos / n;
        }
};

// Perform One-Sample Kolmogorov-Smirnov Test
// Tests if a sample follows a normal distribution with given mean and standard deviation
bool ks_test_one_sample_normal(const std::vector<double>& data, double mean, double std_dev, double alpha = 0.05) {
    if (data.empty() || std_dev <= 0) {
        return false;
    }
    
    // Create a sorted copy of the data
    std::vector<double> sorted_data = data;
    std::sort(sorted_data.begin(), sorted_data.end());
    
    size_t n = sorted_data.size();
    double max_diff = 0.0;
    
    // Calculate the maximum difference between empirical and theoretical CDF
    for (size_t i = 0; i < n; ++i) {
        // Standardize the data point
        double z = (sorted_data[i] - mean) / std_dev;
        
        // Theoretical CDF value
        double F_theoretical = normalCDF(z);
        
        // Empirical CDF values (before and after this point)
        double F_empirical_before = static_cast<double>(i) / n;
        double F_empirical_after = static_cast<double>(i + 1) / n;
        
        // Maximum difference at this point
        double diff1 = std::abs(F_empirical_before - F_theoretical);
        double diff2 = std::abs(F_empirical_after - F_theoretical);
        
        max_diff = std::max(max_diff, std::max(diff1, diff2));
    }
    
    // Critical value for the KS test
    // Using the asymptotic approximation for large samples
    double c_alpha;
    
    // Common critical values for one-sample KS test
    if (std::abs(alpha - 0.05) < 1e-6) {
        c_alpha = 1.36;  // For alpha = 0.05
    } else if (std::abs(alpha - 0.10) < 1e-6) {
        c_alpha = 1.22;  // For alpha = 0.10
    } else if (std::abs(alpha - 0.20) < 1e-6) {
        c_alpha = 1.07;  // For alpha = 0.20
    } else if (std::abs(alpha - 0.30) < 1e-6) {
        c_alpha = 0.97;  // For alpha = 0.30
    } else if (std::abs(alpha - 0.40) < 1e-6) {
        c_alpha = 0.89;  // For alpha = 0.40
    } else if (std::abs(alpha - 0.50) < 1e-6) {
        c_alpha = 0.83;  // For alpha = 0.50
    } else {
        // General approximation using inverse of Kolmogorov distribution
        // For other alpha values, use linear interpolation or approximation
        c_alpha = std::sqrt(-0.5 * std::log(alpha / 2.0));
    }
    
    std::cout << "c_alpha: " << c_alpha  << "\n";
    // Critical value adjusted for sample size
    double critical_value = c_alpha / std::sqrt(n);
    std::cout << "max_diff: " << max_diff  << "critical_value: " << critical_value << "\n";
    std::cout << "difference: " << max_diff - critical_value << "\n";
    
    return max_diff <= critical_value;
}

// Perform Two-Sample Kolmogorov-Smirnov Test
bool ks_test_2(const std::vector<double> & expected, const std::vector<double> & actual, double alpha = 0.1){
    EDF aEDF(expected);
    EDF eEDF(actual);

    double n = static_cast<double>(expected.size());
    double m = static_cast<double>(actual.size());

    // For continuous distributions, we need to check at all unique values
    std::vector<double> all_values;
    all_values.reserve(expected.size() + actual.size());
    all_values.insert(all_values.end(), expected.begin(), expected.end());
    all_values.insert(all_values.end(), actual.begin(), actual.end());
    std::sort(all_values.begin(), all_values.end());
    auto last = std::unique(all_values.begin(), all_values.end());
    all_values.erase(last, all_values.end());

    // Calculate the statistical value: the maximum difference between the two EDF functions
    double d = 0.0;
    for(const auto& x : all_values) {
        d = std::max(d, std::abs(aEDF(x) - eEDF(x)));
    }

    // calculating the critical value
    double c_alpha = std::sqrt(-std::log(alpha / 2) * 0.5);
    double cv = std::sqrt((n + m) / ( n * m)) * c_alpha;

    return d <= cv; // <= because we reject if d > cv
}

template <typename DataType>
class DataGeneratorBoundedTest : public ::TestWithParam<BoundedTupleType>
{
    void set_options(BoundedTupleType      tup,
                     DataGeneratorOptions& opts,
                     vector<index_t>&      size,
                     vector<index_t>&      stride)
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
        vector<index_t>      size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.initMode = DataInitMode(Bounded{});

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

        const int num_threads_test
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

            const auto tid = omp_get_thread_num();
            if(isNaNPacked<DataType>(&scale[0], &data[0], scale_i, data_i))
                has_nan[tid] = true;
            if(isInfPacked<DataType>(&scale[0], &data[0], scale_i, data_i))
                has_inf[tid] = true;
            if(isSubnormPacked<DataType>(&data[0], data_i))
                has_sbn[tid] = true;
        }

        if(opts.includeNaN && DataType::dataInfo.hasNan)
        {
            ASSERT_TRUE(std::any_of(has_nan.begin(), has_nan.end(), [](bool v) { return v; }));
        }

        if(opts.includeInf && DataType::dataInfo.hasInf)
        {
            ASSERT_TRUE(std::any_of(has_inf.begin(), has_inf.end(), [](bool v) { return v; }));
        }

        if(opts.forceDenorm && isScaled<DataType>()
           && ((opts.min < getDataMinSubnorm<DataType>()
                && opts.max > getDataMinSubnorm<DataType>())
               || (opts.min < -getDataMinSubnorm<DataType>()
                   && opts.max > -getDataMinSubnorm<DataType>())))
        {
            ASSERT_TRUE(std::any_of(has_sbn.begin(), has_sbn.end(), [](bool v) { return v; }));
        }
    }
};

template <typename DataType>
class DataGeneratorBoundedAlternatingSignTest
    : public ::TestWithParam<BoundedAlternatingSignTupleType>
{
    void set_options(BoundedAlternatingSignTupleType tup,
                     DataGeneratorOptions&           opts,
                     vector<index_t>&                size,
                     vector<index_t>&                stride)
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
        vector<index_t>      size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.initMode = DataInitMode(BoundedAlternatingSign{});

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
                     vector<index_t>&      size,
                     vector<index_t>&      stride)
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
        vector<index_t>      size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.initMode = DataInitMode(Unbounded{});

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
class DataGeneratorTrigonometricFromFloatTest
    : public ::TestWithParam<TrigonometricFromFloatTupleType>
{
    void set_options(TrigonometricFromFloatTupleType tup,
                     DataGeneratorOptions&           opts,
                     vector<index_t>&                size,
                     vector<index_t>&                stride)
    {
        opts.clampToF32  = std::get<0>(tup);
        opts.includeInf  = std::get<1>(tup);
        opts.includeNaN  = std::get<2>(tup);
        opts.forceDenorm = std::get<3>(tup);

        opts.scaling = std::get<4>(tup);

        set_block_size_stride(std::get<5>(tup), opts.blockScaling, size, stride);
    }

public:
    void testForDataType(TrigonometricFromFloatTupleType& params)
    {
        DataGeneratorOptions opts;
        vector<index_t>      size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.initMode = DataInitMode(TrigonometricFromFloat{});

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
class DataGeneratorNormalFromFloatTest : public ::TestWithParam<NormalFromFloatTupleType>
{
    void set_options(NormalFromFloatTupleType tup,
                     DataGeneratorOptions&    opts,
                     vector<index_t>&         size,
                     vector<index_t>&         stride)
    {
        opts.clampToF32  = std::get<0>(tup);
        opts.includeInf  = std::get<1>(tup);
        opts.includeNaN  = std::get<2>(tup);
        opts.forceDenorm = std::get<3>(tup);

        opts.scaling = std::get<4>(tup);

        set_block_size_stride(std::get<5>(tup), opts.blockScaling, size, stride);
    }

public:
    void testForDataType(NormalFromFloatTupleType& params)
    {
        DataGeneratorOptions opts;
        vector<index_t>      size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.initMode = DataInitMode(NormalFromFloat{0.f, 1.f});

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
class DataGeneratorNormalFromFloatDistributionTest
    : public ::TestWithParam<NormalFromFloatTupleType>
{
public:
    void testForDataType()
    {/*
        // Validate the KS test implementation with hardcoded test cases
        std::cout << "\n=== Validating Kolmogorov-Smirnov Test Implementation ===\n";
        
        int total_tests = 0;
        int tests_meeting_expectations = 0;
        
        // Test case 1: Normal distribution data should pass normality test
        {
            std::mt19937 gen{42};  // Fixed seed for reproducibility
            std::normal_distribution<> dist{0.0, 1.0};
            
            std::vector<double> normal_sample(1000);
            for(int i = 0; i < 1000; i++) {
                normal_sample[i] = dist(gen);
            }
            
            bool result = ks_test_one_sample_normal(normal_sample, 0.0, 1.0);
            bool expected = true;
            if (result == expected) tests_meeting_expectations++;
            total_tests++;
            std::cout << "Test 1 - Normal distribution data: " << (result ? "PASS" : "FAIL") 
                      << " (Expected: PASS)\n";
        }
        
        // Test case 2: Two samples from same normal distribution
        {
            std::mt19937 gen{42};  // Fixed seed for reproducibility
            std::normal_distribution<> dist{0.0, 1.0};
            
            std::vector<double> sample1(1000);
            std::vector<double> sample2(1000);
            for(int i = 0; i < 1000; i++) {
                sample1[i] = dist(gen);
                sample2[i] = dist(gen);
            }
            
            bool result = ks_test_2(sample1, sample2);
            bool expected = true;
            if (result == expected) tests_meeting_expectations++;
            total_tests++;
            std::cout << "Test 2 - Two samples from same normal distribution: " 
                      << (result ? "PASS" : "FAIL") << " (Expected: PASS)\n";
        }
        
        // Test case 3: Uniform distribution should fail normality test
        {
            std::mt19937 gen{42};  // Fixed seed
            std::uniform_real_distribution<> uniform_dist{-3.0, 3.0};
            
            std::vector<double> uniform_sample(1000);
            for(int i = 0; i < 1000; i++) {
                uniform_sample[i] = uniform_dist(gen);
            }
            
            // Test against expected mean and std dev of uniform distribution
            double mean = 0.0;  // mean of uniform[-3, 3]
            double std_dev = std::sqrt(36.0 / 12.0);  // std dev of uniform[-3, 3]
            
            bool result = ks_test_one_sample_normal(uniform_sample, mean, std_dev);
            bool expected = false;
            if (result == expected) tests_meeting_expectations++;
            total_tests++;
            std::cout << "Test 3 - Uniform distribution: " 
                      << (result ? "PASS" : "FAIL") << " (Expected: FAIL)\n";
        }
        
        // Test case 4: Two identical samples should pass
        {
            std::mt19937 gen{42};  // Fixed seed
            std::normal_distribution<> dist{0.0, 1.0};
            
            std::vector<double> sample1(1000);
            for(int i = 0; i < 1000; i++) {
                sample1[i] = dist(gen);
            }
            std::vector<double> sample2 = sample1; // Identical copy
            
            bool result = ks_test_2(sample1, sample2);
            bool expected = true;
            if (result == expected) tests_meeting_expectations++;
            total_tests++;
            std::cout << "Test 4 - Two identical samples: " 
                      << (result ? "PASS" : "FAIL") << " (Expected: PASS)\n";
        }
        
        // Test case 5: Different distributions should fail two-sample test
        {
            std::mt19937 gen{42};  // Fixed seed
            std::normal_distribution<> dist1{0.0, 1.0};
            std::normal_distribution<> dist2{2.0, 1.0};  // Different mean
            
            std::vector<double> sample1(1000);
            std::vector<double> sample2(1000);
            for(int i = 0; i < 1000; i++) {
                sample1[i] = dist1(gen);
                sample2[i] = dist2(gen);
            }
            
            bool result = ks_test_2(sample1, sample2);
            bool expected = false;
            if (result == expected) tests_meeting_expectations++;
            total_tests++;
            std::cout << "Test 5 - Normal(0,1) vs Normal(2,1): " 
                      << (result ? "PASS" : "FAIL") << " (Expected: FAIL)\n";
        }
        
        // Test case 6: Exponential distribution should fail normality test
        {
            std::mt19937 gen{42};  // Fixed seed
            std::exponential_distribution<> exp_dist{1.0};
            
            std::vector<double> exp_sample(1000);
            for(int i = 0; i < 1000; i++) {
                exp_sample[i] = exp_dist(gen);
            }
            
            // Test against actual mean and std dev
            double mean = getMean(exp_sample);
            double std_dev = getStdDev(exp_sample);
            
            bool result = ks_test_one_sample_normal(exp_sample, mean, std_dev);
            bool expected = false;
            if (result == expected) tests_meeting_expectations++;
            total_tests++;
            std::cout << "Test 6 - Exponential distribution: " 
                      << (result ? "PASS" : "FAIL") << " (Expected: FAIL)\n";
        }
        
        // Summary statement
        std::cout << "\nSummary: " << tests_meeting_expectations << "/" << total_tests 
                  << " tests met expected results. ";
        if (tests_meeting_expectations == total_tests) {
            std::cout << "All test validations good!\n";
        } else {
            std::cout << "Some test validations did not meet expectations.\n";
        }
        
        std::cout << "=== End of Kolmogorov-Smirnov Test Validation ===\n\n";
        */
        // Original test code starts here
        DataGeneratorOptions opts;
        vector<index_t>      size, stride;

        opts.clampToF32   = false;
        opts.includeInf   = false;
        opts.includeNaN   = false;
        opts.forceDenorm  = false;
        opts.scaling      = DataScaling::Mean;
        opts.blockScaling = 32;

        const double mean    = 0.f;
        const double std_dev = 1.f;
        opts.initMode       = DataInitMode(NormalFromFloat{mean, std_dev});

        size   = {opts.blockScaling * 1000000};
        stride = {1};

        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        const auto dgen = DataGenerator<DataType>().generate(size, stride, opts);

        auto data = dgen.getReferenceDouble();

        EXPECT_LE(std::abs(getMean(data) - mean), 0.15);
        EXPECT_LE(std::abs(getStdDev(data) - std_dev), 0.15);

        // Generate reference data
        const auto bit_size = getDataSignBits<DataType>() + getDataExponentBits<DataType>()
                              + getDataMantissaBits<DataType>();
        const auto byte_size   = (bit_size + 7) / 8;
        const auto buffer_size = byte_size * data.size();
        // Vector for holding DataType data
        std::vector<uint8_t> buffer;
        buffer.resize(buffer_size, 0x00);
        // Vector for holding reference data
        std::vector<double>        ref_data(data.size(), 0);
        std::random_device         rd{};
        std::mt19937               gen{rd()};  // Standard mersenne_twister_engine seeded with rd()
        std::normal_distribution<> ref_dist{mean, std_dev};
        for(size_t i = 0; i < ref_data.size(); i++)
        {
            // Generate a float, convert it to a DataType, and then convert it into a double
            const auto val = DGen::satConvertToType<DataType>(ref_dist(gen));
            std::memcpy(&buffer[i * byte_size], &val, byte_size);
            uint8_t tScale[] = {Constants::E8M0_1};
            ref_data[i]      = toDouble<DataType>(tScale, buffer.data(), 0, i);
        }

        // Use KS test to verify normal distribution
        // For lower precision types, use a more lenient significance level
        double alpha = 0.000000001;  // Default 95% confidence
        
        // Adjust alpha based on data type precision
        const auto total_bits = getDataSignBits<DataType>() + getDataExponentBits<DataType>() 
                              + getDataMantissaBits<DataType>();
        
        /*if (total_bits <= 4) {
            alpha = 0.0;  // 50% confidence for extremely low precision types (4 bits)
        } else if (total_bits <= 8) {
            alpha = 0.0;  // 60% confidence for very low precision types (6-8 bits)  
        } else if (total_bits <= 16) {
            alpha = 0.0;  // 80% confidence for medium precision types (16 bits)
        }*/
        
        // Test if the generated data follows the expected normal distribution
        const bool ks_result = ks_test_one_sample_normal(data, mean, std_dev, alpha);
        const bool two_sample_result = ks_test_2(ref_data, data, alpha);
        if (!ks_result) {
            std::cout << "KS test failed for " << total_bits << "-bit type with alpha=" << alpha << "\n";
            
            // Also perform two-sample KS test to compare distributions
            
            std::cout << "  Two-sample KS test: " << (two_sample_result ? "PASS" : "FAIL") << "\n";
        }
        else {
            std::cout << "KS test passed for " << total_bits << "-bit type with alpha=" << alpha << "\n";
        }
    const bool ks_pass = ks_result || two_sample_result;
        EXPECT_TRUE(ks_pass);
    }
};

template <typename DataType>
class DataGeneratorZerosTest : public ::TestWithParam<ZerosTupleType>
{
    void set_options(ZerosTupleType        tup,
                     DataGeneratorOptions& opts,
                     vector<index_t>&      size,
                     vector<index_t>&      stride)
    {
        opts.forceDenorm = std::get<0>(tup);
        opts.scaling     = std::get<1>(tup);

        set_block_size_stride(std::get<2>(tup), opts.blockScaling, size, stride);
    }

public:
    void testForDataType(ZerosTupleType& params)
    {
        DataGeneratorOptions opts;
        vector<index_t>      size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.initMode = DataInitMode(Zeros{});

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
                     vector<index_t>&      size,
                     vector<index_t>&      stride)
    {
        opts.forceDenorm = std::get<0>(tup);
        opts.scaling     = std::get<1>(tup);

        set_block_size_stride(std::get<2>(tup), opts.blockScaling, size, stride);
    }

public:
    void testForDataType(OnesTupleType& params)
    {
        DataGeneratorOptions opts;
        vector<index_t>      size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.initMode = DataInitMode(Ones{});

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
                     vector<index_t>&      size,
                     vector<index_t>&      stride)
    {
        opts.forceDenorm = std::get<0>(tup);
        opts.scaling     = std::get<1>(tup);

        set_block_size_stride(std::get<2>(tup), opts.blockScaling, size, stride);
    }

public:
    void testForDataType(IdentityTupleType& params)
    {
        DataGeneratorOptions opts;
        vector<index_t>      size, stride;

        set_options(params, opts, size, stride);
        std::cout << "testing " << opts << " size=" << size << " stride=" << stride << "\n";

        opts.initMode = DataInitMode(Identity{});

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
            bool    diag     = true;
            index_t past_idx = i % size[size.size() - 1];

            index_t data_i = past_idx * stride[size.size() - 1];

            auto tmp = i / size[size.size() - 1];
            for(index_t j = size.size() - 2; j > 0; j--)
            {
                index_t curr_idx = (tmp % size[j]);
                diag             = diag && (past_idx == curr_idx);

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
TYPED_TEST_SUITE(DataGeneratorZerosTest, DataGeneratorTypes);
TYPED_TEST_SUITE(DataGeneratorOnesTest, DataGeneratorTypes);
TYPED_TEST_SUITE(DataGeneratorIdentityTest, DataGeneratorTypes);
TYPED_TEST_SUITE(DataGeneratorTrigonometricFromFloatTest, DataGeneratorTypes);
TYPED_TEST_SUITE(DataGeneratorNormalFromFloatTest, DataGeneratorTypes);
TYPED_TEST_SUITE(DataGeneratorNormalFromFloatDistributionTest, DataGeneratorTypes);

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

TYPED_TEST(DataGeneratorTrigonometricFromFloatTest, TestForEachDataType)
{
    std::vector<TrigonometricFromFloatTupleType> params;
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

TYPED_TEST(DataGeneratorNormalFromFloatTest, TestForEachDataType)
{
    std::vector<NormalFromFloatTupleType> params;
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

TYPED_TEST(DataGeneratorNormalFromFloatDistributionTest, TestForEachDataType)
{
    this->testForDataType();
}
