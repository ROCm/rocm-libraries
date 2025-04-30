// Copyright (c) 2017-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef HIPRAND_HPP_
#define HIPRAND_HPP_

// At least C++11 required
#if defined(__cplusplus) && __cplusplus >= 201103L

#include <random>
#include <exception>
#include <string>
#include <sstream>
#include <type_traits>
#include <limits>

#include "hiprand/hiprand.h"
#include "hiprand/hiprand_kernel.h"

namespace hiprand_cpp {

/// \addtogroup hiprandhostcpp
/// @{

/// \class error
/// \brief A run-time hipRAND error.
///
/// The error class represents an error returned
/// by a hipRAND function.
class error : public std::exception
{
public:
    /// hipRAND error code type
    typedef hiprandStatus_t error_type;

    /// Constructs new error object from error code \p error.
    ///
    /// \param error error code
    error(error_type error) noexcept
        : m_error(error),
          m_error_string(to_string(error))
    {
    }

    ~error() noexcept
    {
    }

    /// Returns the numeric error code.
    error_type error_code() const noexcept
    {
        return m_error;
    }

    /// Returns a string description of the error.
    std::string error_string() const noexcept
    {
        return m_error_string;
    }

    /// Returns a C-string description of the error.
    const char* what() const noexcept
    {
        return m_error_string.c_str();
    }

    /// Static function which converts the numeric hipRAND
    /// error code \p error to a human-readable string.
    ///
    /// If the error code is unknown, a string containing
    /// "Unknown hipRAND error" along with the error code
    /// \p error will be returned.
    static std::string to_string(error_type error)
    {
        switch(error)
        {
            case HIPRAND_STATUS_SUCCESS:
                return "Success";
            case HIPRAND_STATUS_VERSION_MISMATCH:
                return "Header file and linked library version do not match";
            case HIPRAND_STATUS_NOT_INITIALIZED:
                return "Generator was not created using hiprandCreateGenerator";
            case HIPRAND_STATUS_ALLOCATION_FAILED:
                return "Memory allocation failed during execution";
            case HIPRAND_STATUS_TYPE_ERROR:
                return "Generator type is wrong";
            case HIPRAND_STATUS_OUT_OF_RANGE:
                return "Argument out of range";
            case HIPRAND_STATUS_LENGTH_NOT_MULTIPLE:
                return "Length requested is not a multiple of dimension";
            case HIPRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                return "GPU does not have double precision";
            case HIPRAND_STATUS_LAUNCH_FAILURE:
                return "Kernel launch failure";
            case HIPRAND_STATUS_PREEXISTING_FAILURE:
                return "Preexisting failure on library entry";
            case HIPRAND_STATUS_INITIALIZATION_FAILED:
                return "Initialization of HIP failed";
            case HIPRAND_STATUS_ARCH_MISMATCH:
                return "Architecture mismatch, GPU does not support requested feature";
            case HIPRAND_STATUS_INTERNAL_ERROR:
                return "Internal library error";
            case HIPRAND_STATUS_NOT_IMPLEMENTED:
                return "Feature not implemented yet";
            default: {
                std::stringstream s;
                s << "Unknown hipRAND error (" << error << ")";
                return s.str();
            }
        }
    }

    /// Compares two error objects for equality.
    friend
    bool operator==(const error& l, const error& r)
    {
        return l.error_code() == r.error_code();
    }

    /// Compares two error objects for inequality.
    friend
    bool operator!=(const error& l, const error& r)
    {
        return !(l == r);
    }

private:
    error_type m_error;
    std::string m_error_string;
};

/// \class uniform_int_distribution
///
/// \brief Produces random integer values uniformly distributed on the interval [0, 2^(sizeof(IntType)*8) - 1].
///
/// \tparam IntType type of generated values.
/// Only <tt>unsigned char</tt>, <tt>unsigned short</tt>, <tt>unsigned int</tt>, <tt>unsigned long long int</tt> are supported.
template<class IntType = unsigned int>
class uniform_int_distribution
{
    static_assert(std::is_same<unsigned char, IntType>::value
                      || std::is_same<unsigned short, IntType>::value
                      || std::is_same<unsigned int, IntType>::value
                      || std::is_same<unsigned long long int, IntType>::value,
                  "Only unsigned int type is supported in uniform_int_distribution");

public:
    typedef IntType result_type;

    /// Default constructor
    uniform_int_distribution()
    {
    }

    /// Resets distribution's internal state if there is any.
    void reset()
    {
    }

    /// Returns the smallest possible value that can be generated.
    IntType min() const
    {
        return 0;
    }

    /// Returns the largest possible value that can be generated.
    IntType max() const
    {
        return std::numeric_limits<IntType>::max();
    }

    /// \brief Fills \p output with uniformly distributed random integer values.
    ///
    /// Generates \p size random integer values uniformly distributed
    /// on the  interval [0, 2^(sizeof(IntType)*8) - 1], and stores them into the device memory
    /// referenced by \p output pointer.
    ///
    /// \param g An uniform random number generator object
    /// \param output Pointer to device memory to store results
    /// \param size Number of values to generate
    ///
    /// Requirements:
    /// * The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p IntType type.
    /// * If generator \p g is a quasi-random number generator,
    /// then \p size must be a multiple of that generator's dimension.
    ///
    /// See also: hiprandGenerate(), hiprandGenerateChar(), hiprandGenerateShort(), hiprandGenerateLongLong()
    template<class Generator>
    void operator()(Generator& g, IntType * output, size_t size)
    {
        hiprandStatus_t status;
        status = this->generate(g, output, size);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// Returns \c true if the distribution is the same as \p other.
    bool operator==(const uniform_int_distribution<IntType>& other)
    {
        (void) other;
        return true;
    }

    /// Returns \c true if the distribution is different from \p other.
    bool operator!=(const uniform_int_distribution<IntType>& other)
    {
        return !(*this == other);
    }

private:
    template<class Generator>
    hiprandStatus_t generate(Generator& g, unsigned char * output, size_t size)
    {
        return hiprandGenerateChar(g.m_generator, output, size);
    }

    template<class Generator>
    hiprandStatus_t generate(Generator& g, unsigned short * output, size_t size)
    {
        return hiprandGenerateShort(g.m_generator, output, size);
    }

    template<class Generator>
    hiprandStatus_t generate(Generator& g, unsigned int * output, size_t size)
    {
        return hiprandGenerate(g.m_generator, output, size);
    }

    template<class Generator>
    hiprandStatus_t generate(Generator& g, unsigned long long int* output, size_t size)
    {
        return hiprandGenerateLongLong(g.m_generator, output, size);
    }
};

/// \class uniform_real_distribution
///
/// \brief Produces random floating-point values uniformly distributed on the interval (0, 1].
///
/// \tparam RealType type of generated values. Only \p float, \p double and \p half types are supported.
template<class RealType = float>
class uniform_real_distribution
{
    static_assert(
        std::is_same<float, RealType>::value
        || std::is_same<double, RealType>::value
        || std::is_same<half, RealType>::value,
        "Only float, double, and half types are supported in uniform_real_distribution"
    );

public:
    typedef RealType result_type;

    /// Default constructor
    uniform_real_distribution()
    {
    }

    /// Resets distribution's internal state if there is any.
    void reset()
    {
    }

    /// Returns the smallest possible value that can be generated.
    RealType min() const
    {
        if(std::is_same<float, RealType>::value)
        {
            return static_cast<RealType>(2.3283064e-10f);
        }
        return static_cast<RealType>(2.3283064365386963e-10);
    }

    /// Returns the largest possible value that can be generated.
    RealType max() const
    {
        return 1.0;
    }

    /// \brief Fills \p output with uniformly distributed random floating-point values.
    ///
    /// Generates \p size random floating-point values uniformly distributed
    /// on the interval (0, 1], and stores them into the device memory referenced
    /// by \p output pointer.
    ///
    /// \param g An uniform random number generator object
    /// \param output Pointer to device memory to store results
    /// \param size Number of values to generate
    ///
    /// Requirements:
    /// * The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p RealType type.
    /// * If generator \p g is a quasi-random number generator (`hiprand_cpp::sobol32_engine`),
    /// then \p size must be a multiple of that generator's dimension.
    ///
    /// See also: hiprandGenerateUniform(), hiprandGenerateUniformDouble(), hiprandGenerateUniformHalf()
    template<class Generator>
    void operator()(Generator& g, RealType * output, size_t size)
    {
        hiprandStatus_t status;
        status = this->generate(g, output, size);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// Returns \c true if the distribution is the same as \p other.
    bool operator==(const uniform_real_distribution<RealType>& other)
    {
        (void) other;
        return true;
    }

    /// Returns \c true if the distribution is different from \p other.
    bool operator!=(const uniform_real_distribution<RealType>& other)
    {
        return !(*this == other);
    }

private:
    template<class Generator>
    hiprandStatus_t generate(Generator& g, float * output, size_t size)
    {
        return hiprandGenerateUniform(g.m_generator, output, size);
    }

    template<class Generator>
    hiprandStatus_t generate(Generator& g, double * output, size_t size)
    {
        return hiprandGenerateUniformDouble(g.m_generator, output, size);
    }

    template<class Generator>
    hiprandStatus_t generate(Generator& g, half * output, size_t size)
    {
        return hiprandGenerateUniformHalf(g.m_generator, output, size);
    }
};

/// \class normal_distribution
///
/// \brief Produces random numbers according to a normal distribution.
///
/// \tparam RealType type of generated values. Only \p float, \p double and \p half types are supported.
template<class RealType = float>
class normal_distribution
{
    static_assert(
        std::is_same<float, RealType>::value
        || std::is_same<double, RealType>::value
        || std::is_same<half, RealType>::value,
        "Only float, double and half types are supported in normal_distribution"
    );

public:
    typedef RealType result_type;

    /// \class param_type
    /// \brief The type of the distribution parameter set.
    class param_type
    {
    public:
        using distribution_type = normal_distribution<RealType>;
        param_type(RealType mean = 0.0, RealType stddev = 1.0)
            : m_mean(mean), m_stddev(stddev)
        {
        }

        /// \brief Returns the deviation distribution parameter.
        ///
        /// The default value is 1.0.
        RealType mean() const
        {
            return m_mean;
        }

        /// \brief Returns the standard deviation distribution parameter.
        ///
        /// The default value is 1.0.
        RealType stddev() const
        {
            return m_stddev;
        }

        /// Returns \c true if the param_type is the same as \p other.
        bool operator==(const param_type& other)
        {
            return m_mean == other.m_mean && m_stddev == other.m_stddev;
        }

        /// Returns \c true if the param_type is different from \p other.
        bool operator!=(const param_type& other)
        {
            return !(*this == other);
        }
    private:
        RealType m_mean;
        RealType m_stddev;
    };

    /// \brief Constructs a new distribution object.
    /// \param mean A mean distribution parameter
    /// \param stddev A standard deviation distribution parameter
    normal_distribution(RealType mean = 0.0, RealType stddev = 1.0)
        : m_params(mean, stddev)
    {
    }

    /// \brief Constructs a new distribution object.
    /// \param params Distribution parameters
    normal_distribution(const param_type& params)
        : m_params(params)
    {
    }

    /// Resets distribution's internal state if there is any.
    void reset()
    {
    }

    /// \brief Returns the mean distribution parameter.
    ///
    /// The mean specifies the location of the peak. The default value is 0.0.
    RealType mean() const
    {
        return m_params.mean();
    }

    /// \brief Returns the standard deviation distribution parameter.
    ///
    /// The default value is 1.0.
    RealType stddev() const
    {
        return m_params.stddev();
    }

    /// Returns the smallest possible value that can be generated.
    RealType min() const
    {
        return std::numeric_limits<RealType>::lowest();
    }

    /// Returns the largest possible value that can be generated.
    RealType max() const
    {
        return std::numeric_limits<RealType>::max();
    }

    /// Returns the distribution parameter object
    param_type param() const
    {
        return m_params;
    }

    /// Sets the distribution parameter object
    void param(const param_type& params)
    {
        m_params = params;
    }

    /// \brief Fills \p output with normally distributed random floating-point values.
    ///
    /// Generates \p size random floating-point values distributed according to a normal distribution,
    /// and stores them into the device memory referenced by \p output pointer.
    ///
    /// \param g An uniform random number generator object
    /// \param output Pointer to device memory to store results
    /// \param size Number of values to generate
    ///
    /// Requirements:
    /// * The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p RealType type.
    /// * Pointer \p output must be aligned to <tt>2 * sizeof(RealType)</tt> bytes.
    /// * \p size must be even.
    /// * If generator \p g is a quasi-random number generator (`hiprand_cpp::sobol32_engine`),
    /// then \p size must be a multiple of that generator's dimension.
    ///
    /// See also: hiprandGenerateNormal(), hiprandGenerateNormalDouble(), hiprandGenerateNormalHalf()
    template<class Generator>
    void operator()(Generator& g, RealType * output, size_t size)
    {
        hiprandStatus_t status;
        status = this->generate(g, output, size);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \brief Returns \c true if the distribution is the same as \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator==(const normal_distribution<RealType>& other)
    {
        return this->m_params == other.m_params;
    }

    /// \brief Returns \c true if the distribution is different from \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator!=(const normal_distribution<RealType>& other)
    {
        return !(*this == other);
    }

private:
    template<class Generator>
    hiprandStatus_t generate(Generator& g, float * output, size_t size)
    {
        return hiprandGenerateNormal(
            g.m_generator, output, size, this->mean(), this->stddev()
        );
    }

    template<class Generator>
    hiprandStatus_t generate(Generator& g, double * output, size_t size)
    {
        return hiprandGenerateNormalDouble(
            g.m_generator, output, size, this->mean(), this->stddev()
        );
    }

    template<class Generator>
    hiprandStatus_t generate(Generator& g, half * output, size_t size)
    {
        return hiprandGenerateNormalHalf(
            g.m_generator, output, size, this->mean(), this->stddev()
        );
    }

    param_type m_params;
};

/// \class lognormal_distribution
///
/// \brief Produces positive random numbers according to a log-normal distribution.
///
/// \tparam RealType type of generated values. Only \p float, \p double and \p half types are supported.
template<class RealType = float>
class lognormal_distribution
{
    static_assert(
        std::is_same<float, RealType>::value
        || std::is_same<double, RealType>::value
        || std::is_same<half, RealType>::value,
        "Only float, double and half types are supported in lognormal_distribution"
    );

public:
    typedef RealType result_type;

    /// \class param_type
    /// \brief The type of the distribution parameter set.
    class param_type
    {
    public:
        using distribution_type = lognormal_distribution<RealType>;
        param_type(RealType m = 0.0, RealType s = 1.0)
            : m_mean(m), m_stddev(s)
        {
        }

        /// \brief Returns the deviation distribution parameter.
        ///
        /// The default value is 1.0.
        RealType m() const
        {
            return m_mean;
        }

        /// \brief Returns the deviation distribution parameter.
        ///
        /// The default value is 1.0.
        RealType s() const
        {
            return m_stddev;
        }

        /// Returns \c true if the param_type is the same as \p other.
        bool operator==(const param_type& other)
        {
            return m_mean == other.m_mean && m_stddev == other.m_stddev;
        }

        /// Returns \c true if the param_type is different from \p other.
        bool operator!=(const param_type& other)
        {
            return !(*this == other);
        }
    private:
        RealType m_mean;
        RealType m_stddev;
    };

    /// \brief Constructs a new distribution object.
    /// \param m A mean distribution parameter
    /// \param s A standard deviation distribution parameter
    lognormal_distribution(RealType m = 0.0, RealType s = 1.0)
        : m_params(m, s)
    {
    }

    /// \brief Constructs a new distribution object.
    /// \param params Distribution parameters
    lognormal_distribution(const param_type& params)
        : m_params(params)
    {
    }

    /// Resets distribution's internal state if there is any.
    void reset()
    {
    }

    /// \brief Returns the mean distribution parameter.
    ///
    /// The mean specifies the location of the peak. The default value is 0.0.
    RealType m() const
    {
        return m_params.m();
    }

    /// \brief Returns the standard deviation distribution parameter.
    ///
    /// The default value is 1.0.
    RealType s() const
    {
        return m_params.s();
    }

    /// Returns the distribution parameter object
    param_type param() const
    {
        return m_params;
    }

    /// Sets the distribution parameter object
    void param(const param_type& params)
    {
        m_params = params;
    }

    /// Returns the smallest possible value that can be generated.
    RealType min() const
    {
        return 0;
    }

    /// Returns the largest possible value that can be generated.
    RealType max() const
    {
        return std::numeric_limits<RealType>::max();
    }

    /// \brief Fills \p output with log-normally distributed random floating-point values.
    ///
    /// Generates \p size random floating-point values (greater than zero) distributed according
    /// to a log-normal distribution, and stores them into the device memory referenced
    /// by \p output pointer.
    ///
    /// \param g An uniform random number generator object
    /// \param output Pointer to device memory to store results
    /// \param size Number of values to generate
    ///
    /// Requirements:
    /// * The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p RealType type.
    /// * Pointer \p output must be aligned to <tt>2 * sizeof(RealType)</tt> bytes.
    /// * \p size must be even.
    /// * If generator \p g is a quasi-random number generator (`hiprand_cpp::sobol32_engine`),
    /// then \p size must be a multiple of that generator's dimension.
    ///
    /// See also: hiprandGenerateLogNormal(), hiprandGenerateLogNormalDouble(), hiprandGenerateLogNormalHalf()
    template<class Generator>
    void operator()(Generator& g, RealType * output, size_t size)
    {
        hiprandStatus_t status;
        status = this->generate(g, output, size);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \brief Returns \c true if the distribution is the same as \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator==(const lognormal_distribution<RealType>& other)
    {
        return this->m_params == other.m_params;
    }

    /// \brief Returns \c true if the distribution is different from \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator!=(const lognormal_distribution<RealType>& other)
    {
        return !(*this == other);
    }

private:
    template<class Generator>
    hiprandStatus_t generate(Generator& g, float * output, size_t size)
    {
        return hiprandGenerateLogNormal(
            g.m_generator, output, size, this->m(), this->s()
        );
    }

    template<class Generator>
    hiprandStatus_t generate(Generator& g, double * output, size_t size)
    {
        return hiprandGenerateLogNormalDouble(
            g.m_generator, output, size, this->m(), this->s()
        );
    }

    template<class Generator>
    hiprandStatus_t generate(Generator& g, half * output, size_t size)
    {
        return hiprandGenerateLogNormalHalf(
            g.m_generator, output, size, this->m(), this->s()
        );
    }

    param_type m_params;
};

/// \class poisson_distribution
///
/// \brief Produces random non-negative integer values distributed according to Poisson distribution.
///
/// \tparam IntType type of generated values. Only \p unsinged \p int type is supported.
template<class IntType = unsigned int>
class poisson_distribution
{
    static_assert(
        std::is_same<unsigned int, IntType>::value,
        "Only unsigned int type is supported in poisson_distribution"
    );

public:
    typedef IntType result_type;

    /// \class param_type
    /// \brief The type of the distribution parameter set.
    class param_type
    {
    public:
        using distribution_type = poisson_distribution<IntType>;
        param_type(double mean = 1.0)
            : m_mean(mean)
        {
        }

        /// \brief Returns the mean distribution parameter.
        ///
        /// The mean (also known as lambda) is the average number
        /// of events per interval. The default value is 1.0.
        double mean() const
        {
            return m_mean;
        }

        /// Returns \c true if the param_type is the same as \p other.
        bool operator==(const param_type& other)
        {
            return m_mean == other.m_mean;
        }

        /// Returns \c true if the param_type is different from \p other.
        bool operator!=(const param_type& other)
        {
            return !(*this == other);
        }

    private:
        double m_mean;
    };

    /// \brief Constructs a new distribution object.
    /// \param mean A mean distribution parameter.
    poisson_distribution(double mean = 1.0)
        : m_params(mean)
    {
    }

    /// \brief Constructs a new distribution object.
    /// \param params Distribution parameters
    poisson_distribution(const param_type& params)
        : m_params(params)
    {
    }

    /// Resets distribution's internal state if there is any.
    void reset()
    {
    }

    /// \brief Returns the mean distribution parameter.
    ///
    /// The mean (also known as lambda) is the average number
    /// of events per interval. The default value is 1.0.
    double mean() const
    {
        return m_params.mean();
    }

    /// Returns the smallest possible value that can be generated.
    IntType min() const
    {
        return 0;
    }

    /// Returns the largest possible value that can be generated.
    IntType max() const
    {
        return std::numeric_limits<IntType>::max();
    }

    /// Returns the distribution parameter object
    param_type param() const
    {
        return m_params;
    }

    /// Sets the distribution parameter object
    void param(const param_type& params)
    {
        m_params = params;
    }

    /// \brief Fills \p output with random non-negative integer values
    /// distributed according to Poisson distribution.
    ///
    /// Generates \p size random non-negative integer values distributed according
    /// to Poisson distribution, and stores them into the device memory referenced
    /// by \p output pointer.
    ///
    /// \param g An uniform random number generator object
    /// \param output Pointer to device memory to store results
    /// \param size Number of values to generate
    ///
    /// Requirements:
    /// * The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p IntType type.
    /// * If generator \p g is a quasi-random number generator (`hiprand_cpp::sobol32_engine`),
    /// then \p size must be a multiple of that generator's dimension.
    ///
    /// See also: hiprandGeneratePoisson()
    template<class Generator>
    void operator()(Generator& g, IntType * output, size_t size)
    {
        hiprandStatus_t status;
        status = hiprandGeneratePoisson(g.m_generator, output, size, this->mean());
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \brief Returns \c true if the distribution is the same as \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator==(const poisson_distribution<IntType>& other)
    {
        return this->m_params == other.m_params;
    }

    /// \brief Returns \c true if the distribution is different from \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator!=(const poisson_distribution<IntType>& other)
    {
        return !(*this == other);
    }

private:
    param_type m_params;
};

/// \brief Pseudorandom number engine based Philox algorithm.
///
/// philox4x32_10_engine implements
/// a Counter-based random number generator called Philox, which was developed by
/// a group at D. E. Shaw Research.
/// It generates random numbers of type \p unsigned \p int on the interval [0; 2^32 - 1].
/// Random numbers are generated in sets of four.
template<unsigned long long DefaultSeed = HIPRAND_PHILOX4x32_DEFAULT_SEED>
class philox4x32_10_engine
{
public:
    /// \typedef result_type
    /// Type of values generated by the random number engine.
    typedef unsigned int result_type;
    // \typedef order_type
    /// Pseudo-random number engine ordering type.
    /// Represents the ordering of the results of a random number engine.
    ///
    /// See also: order()
    typedef hiprandOrdering_t order_type;
    /// \typedef offset_type
    /// Pseudo-random number engine offset type.
    /// Offset represents a number of the random number engine's states
    /// that should be skipped before first value is generated.
    ///
    /// See also: offset()
    typedef unsigned long long offset_type;
    /// \typedef seed_type
    /// Pseudo-random number engine seed type definition.
    ///
    /// See also: seed()
    typedef unsigned long long seed_type;
    /// \brief The default seed equal to \p DefaultSeed.
    static constexpr seed_type default_seed = DefaultSeed;

    /// \brief Constructs the pseudo-random number engine.
    ///
    /// \param seed_value seed value to use in the initialization of the internal state, see also seed()
    /// \param offset_value number of internal states that should be skipped, see also offset()
    ///
    /// See also: hiprandCreateGenerator()
    philox4x32_10_engine(seed_type seed_value = DefaultSeed,
                         offset_type offset_value = 0)
    {
        hiprandStatus_t status;
        status = hiprandCreateGenerator(&m_generator, this->type());
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
        try
        {
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)hiprandDestroyGenerator(m_generator);
            throw;
        }
    }

    /// \brief Constructs the pseudo-random number engine.
    ///
    /// The pseudo-random number engine will be created using \p generator.
    /// The constructed engine take ownership over \p generator, and sets
    /// passed reference to \p NULL. The lifetime of \p generator is now
    /// bound to the lifetime of the engine.
    ///
    /// \param generator hipRAND generator
    philox4x32_10_engine(hiprandGenerator_t& generator)
        : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw hiprand_cpp::error(HIPRAND_STATUS_NOT_INITIALIZED);
        }
        generator = NULL;
    }

    philox4x32_10_engine(const philox4x32_10_engine&) = delete;

    philox4x32_10_engine(philox4x32_10_engine&&) = delete;

    philox4x32_10_engine& operator=(const philox4x32_10_engine&) = delete;

    philox4x32_10_engine& operator=(philox4x32_10_engine&&) = delete;

    /// Destructs the engine.
    ///
    /// See also: hiprandDestroyGenerator()
    ~philox4x32_10_engine() noexcept(false)
    {
        hiprandStatus_t status = hiprandDestroyGenerator(m_generator);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \brief Sets the random number engine's \p hipStream for kernel launches.
    /// \param value new \p hipStream to use
    void stream(hipStream_t value)
    {
        hiprandStatus_t status = hiprandSetStream(m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \brief Sets the offset of a random number engine.
    ///
    /// Offset represents a number of the random number engine's states
    /// that should be skipped before first value is generated.
    ///
    /// - This operation resets the engine's internal state.
    /// - This operation does not change the engine's seed or the number of dimensions.
    ///
    /// \param value New absolute offset
    ///
    /// See also: hiprandSetGeneratorOffset()
    void offset(offset_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOffset(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \brief Sets the order of a random number engine.
    ///
    /// The order refers to the ordering of the sequences generated
    /// by the engine.
    ///
    /// - This operation resets the engine's internal state.
    /// - This operation does not change the engine's seed.
    ///
    /// \param value New ordering
    ///
    /// See also: hiprandSetGeneratorOrdering()
    void order(order_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOrdering(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \brief Sets the seed of the pseudo-random number engine.
    ///
    /// - This operation resets the engine's internal state.
    /// - This operation does not change the engine's offset.
    ///
    /// \param value New seed value
    ///
    /// See also: hiprandSetPseudoRandomGeneratorSeed()
    void seed(seed_type value)
    {
        hiprandStatus_t status = hiprandSetPseudoRandomGeneratorSeed(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \brief Fills \p output with uniformly distributed random integer values.
    ///
    /// Generates \p size random integer values uniformly distributed
    /// on the interval [0, 2^32 - 1], and stores them into the device memory
    /// referenced by \p output pointer.
    ///
    /// \param output Pointer to device memory to store results
    /// \param size Number of values to generate
    ///
    /// The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p IntType type.
    ///
    /// See also: hiprandGenerate()
    template<class Generator>
    void operator()(result_type * output, size_t size)
    {
        hiprandStatus_t status;
        status = hiprandGenerate(m_generator, output, size);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// Returns the smallest possible value that can be generated by the engine.
    result_type min() const
    {
        return 0;
    }

    /// Returns the largest possible value that can be generated by the engine.
    result_type max() const
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// Returns type of the hipRAND pseudo-random number generator associated with the engine.
    static constexpr hiprandRngType type()
    {
        return HIPRAND_RNG_PSEUDO_PHILOX4_32_10;
    }

private:
    hiprandGenerator_t m_generator;

    /// \cond
    template<class T>
    friend class ::hiprand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::hiprand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::hiprand_cpp::normal_distribution;

    template<class T>
    friend class ::hiprand_cpp::lognormal_distribution;

    template<class T>
    friend class ::hiprand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename philox4x32_10_engine<DefaultSeed>::seed_type philox4x32_10_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Pseudorandom number engine based XORWOW algorithm.
///
/// xorwow_engine is a xorshift pseudorandom
/// number engine based on XORWOW algorithm, which was presented by George Marsaglia in
/// "Xorshift RNGs" paper published in Journal of Statistical Software. It produces random numbers
/// of type \p unsigned \p int on the interval [0; 2^32 - 1].
template<unsigned long long DefaultSeed = HIPRAND_XORWOW_DEFAULT_SEED>
class xorwow_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef hiprandOrdering_t order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(seed_type, offset_type)
    xorwow_engine(seed_type seed_value = DefaultSeed,
                  offset_type offset_value = 0)
    {
        hiprandStatus_t status;
        status = hiprandCreateGenerator(&m_generator, this->type());
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
        try
        {
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)hiprandDestroyGenerator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(hiprandGenerator_t&)
    xorwow_engine(hiprandGenerator_t& generator)
        : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw hiprand_cpp::error(HIPRAND_STATUS_NOT_INITIALIZED);
        }
        generator = NULL;
    }

    xorwow_engine(const xorwow_engine&) = delete;

    xorwow_engine(xorwow_engine&&) = delete;

    xorwow_engine& operator=(const xorwow_engine&) = delete;

    xorwow_engine& operator=(xorwow_engine&&) = delete;

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~xorwow_engine() noexcept(false)
    {
        hiprandStatus_t status = hiprandDestroyGenerator(m_generator);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        hiprandStatus_t status = hiprandSetStream(m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOffset(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOrdering(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        hiprandStatus_t status = hiprandSetPseudoRandomGeneratorSeed(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type * output, size_t size)
    {
        hiprandStatus_t status;
        status = hiprandGenerate(m_generator, output, size);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    result_type min() const
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    result_type max() const
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr hiprandRngType type()
    {
        return HIPRAND_RNG_PSEUDO_XORWOW;
    }

private:
    hiprandGenerator_t m_generator;

    /// \cond
    template<class T>
    friend class ::hiprand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::hiprand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::hiprand_cpp::normal_distribution;

    template<class T>
    friend class ::hiprand_cpp::lognormal_distribution;

    template<class T>
    friend class ::hiprand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename xorwow_engine<DefaultSeed>::seed_type xorwow_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Pseudorandom number engine based MRG32k3a CMRG.
///
/// mrg32k3a_engine is an implementation of MRG32k3a pseudorandom number generator,
/// which is a Combined Multiple Recursive Generator (CMRG) created by Pierre L'Ecuyer.
/// It produces random 32-bit \p unsigned \p int values on the interval [0; 2^32 - 1].
template<unsigned long long DefaultSeed = HIPRAND_MRG32K3A_DEFAULT_SEED>
class mrg32k3a_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef hiprandOrdering_t order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(seed_type, offset_type)
    mrg32k3a_engine(seed_type seed_value = DefaultSeed,
                    offset_type offset_value = 0)
    {
        hiprandStatus_t status;
        status = hiprandCreateGenerator(&m_generator, this->type());
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
        try
        {
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)hiprandDestroyGenerator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(hiprandGenerator_t&)
    mrg32k3a_engine(hiprandGenerator_t& generator)
        : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw hiprand_cpp::error(HIPRAND_STATUS_NOT_INITIALIZED);
        }
        generator = NULL;
    }

    mrg32k3a_engine(const mrg32k3a_engine&) = delete;

    mrg32k3a_engine(mrg32k3a_engine&&) = delete;

    mrg32k3a_engine& operator=(const mrg32k3a_engine&) = delete;

    mrg32k3a_engine& operator=(mrg32k3a_engine&&) = delete;

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~mrg32k3a_engine() noexcept(false)
    {
        hiprandStatus_t status = hiprandDestroyGenerator(m_generator);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        hiprandStatus_t status = hiprandSetStream(m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOffset(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOrdering(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        hiprandStatus_t status = hiprandSetPseudoRandomGeneratorSeed(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type * output, size_t size)
    {
        hiprandStatus_t status;
        status = hiprandGenerate(m_generator, output, size);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    result_type min() const
    {
        return 1;
    }

    /// \copydoc philox4x32_10_engine::max()
    result_type max() const
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr hiprandRngType type()
    {
        return HIPRAND_RNG_PSEUDO_MRG32K3A;
    }

private:
    hiprandGenerator_t m_generator;

    /// \cond
    template<class T>
    friend class ::hiprand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::hiprand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::hiprand_cpp::normal_distribution;

    template<class T>
    friend class ::hiprand_cpp::lognormal_distribution;

    template<class T>
    friend class ::hiprand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename mrg32k3a_engine<DefaultSeed>::seed_type mrg32k3a_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Pseudorandom number engine based on Mersenne Twister
/// for Graphic Processors algorithm.
///
/// mtgp32_engine is a random number engine based on Mersenne Twister
/// for Graphic Processors algorithm, which is a version of well-known Mersenne Twister
/// algorithm. It produces high quality random numbers of type \p unsigned \p int
/// on the interval [0; 2^32 - 1].
template<unsigned long long DefaultSeed = HIPRAND_MTGP32_DEFAULT_SEED>
class mtgp32_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef hiprandOrdering_t order_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \brief Constructs the pseudo-random number engine.
    ///
    /// MTGP32 engine does not accept offset.
    ///
    /// \param seed_value seed value to use in the initialization of the internal state, see also seed()
    ///
    /// See also: hiprandCreateGenerator()
    mtgp32_engine(seed_type seed_value = DefaultSeed)
    {
        hiprandStatus_t status;
        status = hiprandCreateGenerator(&m_generator, this->type());
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
        try
        {
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)hiprandDestroyGenerator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(hiprandGenerator_t&)
    mtgp32_engine(hiprandGenerator_t& generator)
        : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw hiprand_cpp::error(HIPRAND_STATUS_NOT_INITIALIZED);
        }
        generator = NULL;
    }

    mtgp32_engine(const mtgp32_engine&) = delete;

    mtgp32_engine(mtgp32_engine&&) = delete;

    mtgp32_engine& operator=(const mtgp32_engine&) = delete;

    mtgp32_engine& operator=(mtgp32_engine&&) = delete;

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~mtgp32_engine() noexcept(false)
    {
        hiprandStatus_t status = hiprandDestroyGenerator(m_generator);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        hiprandStatus_t status = hiprandSetStream(m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOrdering(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        hiprandStatus_t status = hiprandSetPseudoRandomGeneratorSeed(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type * output, size_t size)
    {
        hiprandStatus_t status;
        status = hiprandGenerate(m_generator, output, size);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    result_type min() const
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    result_type max() const
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr hiprandRngType type()
    {
        return HIPRAND_RNG_PSEUDO_MTGP32;
    }

private:
    hiprandGenerator_t m_generator;

    /// \cond
    template<class T>
    friend class ::hiprand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::hiprand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::hiprand_cpp::normal_distribution;

    template<class T>
    friend class ::hiprand_cpp::lognormal_distribution;

    template<class T>
    friend class ::hiprand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename mtgp32_engine<DefaultSeed>::seed_type mtgp32_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Pseudorandom number engine based on Mersenne Twister.
///
/// mt19937_engine is a random number engine based on the well-known
/// Mersenne Twister algorithm. It produces high quality random numbers of
/// type \p unsigned \p int on the interval [0; 2^32 - 1].
template<unsigned long long DefaultSeed = HIPRAND_MT19937_DEFAULT_SEED>
class mt19937_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef hiprandOrdering_t order_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \brief Constructs the pseudo-random number engine.
    ///
    /// MT19937 engine does not accept offset.
    ///
    /// \param seed_value seed value to use in the initialization of the internal state, see also seed()
    ///
    /// See also: hiprandCreateGenerator()
    mt19937_engine(seed_type seed_value = DefaultSeed)
    {
        hiprandStatus_t status;
        status = hiprandCreateGenerator(&m_generator, this->type());
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
        try
        {
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)hiprandDestroyGenerator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(hiprandGenerator_t&)
    mt19937_engine(hiprandGenerator_t& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw hiprand_cpp::error(HIPRAND_STATUS_NOT_INITIALIZED);
        }
        generator = NULL;
    }

    mt19937_engine(const mt19937_engine&) = delete;

    mt19937_engine(mt19937_engine&&) = delete;

    mt19937_engine& operator=(const mt19937_engine&) = delete;

    mt19937_engine& operator=(mt19937_engine&&) = delete;

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~mt19937_engine() noexcept(false)
    {
        hiprandStatus_t status = hiprandDestroyGenerator(m_generator);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        hiprandStatus_t status = hiprandSetStream(m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOrdering(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        hiprandStatus_t status = hiprandSetPseudoRandomGeneratorSeed(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        hiprandStatus_t status;
        status = hiprandGenerate(m_generator, output, size);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    result_type min() const
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    result_type max() const
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr hiprandRngType type()
    {
        return HIPRAND_RNG_PSEUDO_MT19937;
    }

private:
    hiprandGenerator_t m_generator;

    /// \cond
    template<class T>
    friend class ::hiprand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::hiprand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::hiprand_cpp::normal_distribution;

    template<class T>
    friend class ::hiprand_cpp::lognormal_distribution;

    template<class T>
    friend class ::hiprand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename mt19937_engine<DefaultSeed>::seed_type mt19937_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Sobol's quasi-random sequence generator
///
/// sobol32_engine is quasi-random number engine which produced Sobol sequences.
/// This implementation supports generating sequences in up to 20,000 dimensions.
/// The engine produces random unsigned integers on the interval [0; 2^32 - 1].
template<unsigned int DefaultNumDimensions = 1>
class sobol32_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef hiprandOrdering_t order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \typedef dimensions_num_type
    /// Quasi-random number engine type for number of dimensions.
    ///
    /// See also dimensions()
    typedef unsigned int dimensions_num_type;
    /// \brief The default number of dimenstions, equal to \p DefaultNumDimensions.
    static constexpr dimensions_num_type default_num_dimensions = DefaultNumDimensions;

    /// \brief Constructs the pseudo-random number engine.
    ///
    /// \param num_of_dimensions number of dimensions to use in the initialization of the internal state, see also dimensions()
    /// \param offset_value number of internal states that should be skipped, see also offset()
    ///
    /// See also: hiprandCreateGenerator()
    sobol32_engine(dimensions_num_type num_of_dimensions = DefaultNumDimensions,
                   offset_type offset_value = 0)
    {
        hiprandStatus_t status;
        status = hiprandCreateGenerator(&m_generator, this->type());
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
        try
        {
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->dimensions(num_of_dimensions);
        }
        catch(...)
        {
            (void)hiprandDestroyGenerator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(hiprandGenerator_t&)
    sobol32_engine(hiprandGenerator_t& generator)
        : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw hiprand_cpp::error(HIPRAND_STATUS_NOT_INITIALIZED);
        }
        generator = NULL;
    }

    sobol32_engine(const sobol32_engine&) = delete;

    sobol32_engine(sobol32_engine&&) = delete;

    sobol32_engine& operator=(const sobol32_engine&) = delete;

    sobol32_engine& operator=(sobol32_engine&&) = delete;

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~sobol32_engine() noexcept(false)
    {
        hiprandStatus_t status = hiprandDestroyGenerator(m_generator);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        hiprandStatus_t status = hiprandSetStream(m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOffset(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOrdering(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \brief Set the number of dimensions of a quasi-random number generator.
    ///
    /// Supported values of \p dimensions are 1 to 20000.
    ///
    /// - This operation resets the generator's internal state.
    /// - This operation does not change the generator's offset.
    ///
    /// \param value Number of dimensions
    ///
    /// See also: hiprandSetQuasiRandomGeneratorDimensions()
    void dimensions(dimensions_num_type value)
    {
        hiprandStatus_t status =
            hiprandSetQuasiRandomGeneratorDimensions(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \brief Fills \p output with uniformly distributed random integer values.
    ///
    /// Generates \p size random integer values uniformly distributed
    /// on the interval [0, 2^32 - 1], and stores them into the device memory
    /// referenced by \p output pointer.
    ///
    /// \param output Pointer to device memory to store results
    /// \param size Number of values to generate
    ///
    /// Requirements:
    /// * The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p IntType type.
    /// * \p size must be a multiple of the engine's number of dimensions.
    ////
    /// See also: hiprandGenerate()
    template<class Generator>
    void operator()(result_type * output, size_t size)
    {
        hiprandStatus_t status;
        status = hiprandGenerate(m_generator, output, size);
        if(status != HIPRAND_STATUS_SUCCESS) throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    result_type min() const
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    result_type max() const
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr hiprandRngType type()
    {
        return HIPRAND_RNG_QUASI_SOBOL32;
    }

private:
    hiprandGenerator_t m_generator;

    /// \cond
    template<class T>
    friend class ::hiprand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::hiprand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::hiprand_cpp::normal_distribution;

    template<class T>
    friend class ::hiprand_cpp::lognormal_distribution;

    template<class T>
    friend class ::hiprand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned int DefaultNumDimensions>
constexpr typename sobol32_engine<DefaultNumDimensions>::dimensions_num_type
sobol32_engine<DefaultNumDimensions>::default_num_dimensions;
/// \endcond

/// \brief Sobol's quasi-random sequence generator
///
/// scrambled_sobol32_engine is a quasi-random number engine which produces scrambled Sobol sequences.
/// This implementation supports generating sequences in up to 20,000 dimensions.
/// The engine produces random unsigned integers on the interval [0; 2^32 - 1].
template<unsigned int DefaultNumDimensions = 1>
class scrambled_sobol32_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef hiprandOrdering_t order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc sobol32_engine::dimensions_num_type
    typedef unsigned int dimensions_num_type;
    /// \copydoc sobol32_engine::default_num_dimensions
    static constexpr dimensions_num_type default_num_dimensions = DefaultNumDimensions;

    /// \copydoc sobol32_engine::sobol32_engine(dimensions_num_type, offset_type)
    scrambled_sobol32_engine(dimensions_num_type num_of_dimensions = DefaultNumDimensions,
                             offset_type         offset_value      = 0)
    {
        hiprandStatus_t status;
        status = hiprandCreateGenerator(&m_generator, this->type());
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
        try
        {
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->dimensions(num_of_dimensions);
        }
        catch(...)
        {
            (void)hiprandDestroyGenerator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(hiprandGenerator_t&)
    scrambled_sobol32_engine(hiprandGenerator_t& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw hiprand_cpp::error(HIPRAND_STATUS_NOT_INITIALIZED);
        }
        generator = NULL;
    }

    scrambled_sobol32_engine(const scrambled_sobol32_engine&) = delete;

    scrambled_sobol32_engine(scrambled_sobol32_engine&&) = delete;

    scrambled_sobol32_engine& operator=(const scrambled_sobol32_engine&) = delete;

    scrambled_sobol32_engine& operator=(scrambled_sobol32_engine&&) = delete;

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~scrambled_sobol32_engine() noexcept(false)
    {
        hiprandStatus_t status = hiprandDestroyGenerator(m_generator);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        hiprandStatus_t status = hiprandSetStream(m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOffset(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOrdering(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc sobol32_engine::dimensions()
    void dimensions(dimensions_num_type value)
    {
        hiprandStatus_t status = hiprandSetQuasiRandomGeneratorDimensions(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc sobol32::operator()(result_type, size_t)
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        hiprandStatus_t status;
        status = hiprandGenerate(m_generator, output, size);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    result_type min() const
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    result_type max() const
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr hiprandRngType type()
    {
        return HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
    }

private:
    hiprandGenerator_t m_generator;

    /// \cond
    template<class T>
    friend class ::hiprand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::hiprand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::hiprand_cpp::normal_distribution;

    template<class T>
    friend class ::hiprand_cpp::lognormal_distribution;

    template<class T>
    friend class ::hiprand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned int DefaultNumDimensions>
constexpr typename scrambled_sobol32_engine<DefaultNumDimensions>::dimensions_num_type
    scrambled_sobol32_engine<DefaultNumDimensions>::default_num_dimensions;
/// \endcond

/// \brief Sobol's quasi-random sequence generator
///
/// sobol64_engine is a quasi-random number engine which produced Sobol sequences.
/// This implementation supports generating sequences in up to 20,000 dimensions.
/// The engine produces random unsigned integers on the interval [0; 2^64 - 1].
template<unsigned int DefaultNumDimensions = 1>
class sobol64_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned long long int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef hiprandOrdering_t order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc sobol32_engine::dimensions_num_type
    typedef unsigned int dimensions_num_type;
    /// \copydoc sobol32_engine::default_num_dimensions
    static constexpr dimensions_num_type default_num_dimensions = DefaultNumDimensions;

    /// \copydoc sobol32_engine::sobol32_engine(dimensions_num_type, offset_type)
    sobol64_engine(dimensions_num_type num_of_dimensions = DefaultNumDimensions,
                   offset_type         offset_value      = 0)
    {
        hiprandStatus_t status;
        status = hiprandCreateGenerator(&m_generator, this->type());
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
        try
        {
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->dimensions(num_of_dimensions);
        }
        catch(...)
        {
            (void)hiprandDestroyGenerator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(hiprandGenerator_t&)
    sobol64_engine(hiprandGenerator_t& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw hiprand_cpp::error(HIPRAND_STATUS_NOT_INITIALIZED);
        }
        generator = NULL;
    }

    sobol64_engine(const sobol64_engine&) = delete;

    sobol64_engine(sobol64_engine&&) = delete;

    sobol64_engine& operator=(const sobol64_engine&) = delete;

    sobol64_engine& operator=(sobol64_engine&&) = delete;

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~sobol64_engine() noexcept(false)
    {
        hiprandStatus_t status = hiprandDestroyGenerator(m_generator);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        hiprandStatus_t status = hiprandSetStream(m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOffset(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOrdering(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc sobol32_engine::dimensions()
    void dimensions(dimensions_num_type value)
    {
        hiprandStatus_t status = hiprandSetQuasiRandomGeneratorDimensions(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \brief Fills \p output with uniformly distributed random integer values.
    ///
    /// Generates \p size random integer values uniformly distributed
    /// on the interval [0, 2^64 - 1], and stores them into the device memory
    /// referenced by \p output pointer.
    ///
    /// \param output Pointer to device memory to store results
    /// \param size Number of values to generate
    ///
    /// Requirements:
    /// * The device memory pointed by \p output must have been previously allocated
    /// and be large enough to store at least \p size values of \p IntType type.
    /// * \p size must be a multiple of the engine's number of dimensions.
    ////
    /// See also: hiprandGenerate()
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        hiprandStatus_t status;
        status = hiprandGenerateLongLong(m_generator, output, size);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    result_type min() const
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    result_type max() const
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr hiprandRngType type()
    {
        return HIPRAND_RNG_QUASI_SOBOL64;
    }

private:
    hiprandGenerator_t m_generator;

    /// \cond
    template<class T>
    friend class ::hiprand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::hiprand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::hiprand_cpp::normal_distribution;

    template<class T>
    friend class ::hiprand_cpp::lognormal_distribution;

    template<class T>
    friend class ::hiprand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned int DefaultNumDimensions>
constexpr typename sobol64_engine<DefaultNumDimensions>::dimensions_num_type
    sobol64_engine<DefaultNumDimensions>::default_num_dimensions;
/// \endcond

/// \brief Sobol's quasi-random sequence generator
///
/// scrambled_sobol64_engine is a quasi-random number engine which produces scrambled Sobol sequences.
/// This implementation supports generating sequences in up to 20,000 dimensions.
/// The engine produces random unsigned integers on the interval [0; 2^64 - 1].
template<unsigned int DefaultNumDimensions = 1>
class scrambled_sobol64_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned long long int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef hiprandOrdering_t order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc sobol32_engine::dimensions_num_type
    typedef unsigned int dimensions_num_type;
    /// \copydoc sobol32_engine::default_num_dimensions
    static constexpr dimensions_num_type default_num_dimensions = DefaultNumDimensions;

    /// \copydoc sobol32_engine::sobol32_engine(dimensions_num_type, offset_type)
    scrambled_sobol64_engine(dimensions_num_type num_of_dimensions = DefaultNumDimensions,
                             offset_type         offset_value      = 0)
    {
        hiprandStatus_t status;
        status = hiprandCreateGenerator(&m_generator, this->type());
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
        try
        {
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->dimensions(num_of_dimensions);
        }
        catch(...)
        {
            (void)hiprandDestroyGenerator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(hiprandGenerator_t&)
    scrambled_sobol64_engine(hiprandGenerator_t& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw hiprand_cpp::error(HIPRAND_STATUS_NOT_INITIALIZED);
        }
        generator = NULL;
    }

    scrambled_sobol64_engine(const scrambled_sobol64_engine&) = delete;

    scrambled_sobol64_engine(scrambled_sobol64_engine&&) = delete;

    scrambled_sobol64_engine& operator=(const scrambled_sobol64_engine&) = delete;

    scrambled_sobol64_engine& operator=(scrambled_sobol64_engine&&) = delete;

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~scrambled_sobol64_engine() noexcept(false)
    {
        hiprandStatus_t status = hiprandDestroyGenerator(m_generator);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        hiprandStatus_t status = hiprandSetStream(m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOffset(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        hiprandStatus_t status = hiprandSetGeneratorOrdering(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc sobol32_engine::dimensions()
    void dimensions(dimensions_num_type value)
    {
        hiprandStatus_t status = hiprandSetQuasiRandomGeneratorDimensions(this->m_generator, value);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc sobol32::operator()(result_type, size_t)
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        hiprandStatus_t status;
        status = hiprandGenerateLongLong(m_generator, output, size);
        if(status != HIPRAND_STATUS_SUCCESS)
            throw hiprand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    result_type min() const
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    result_type max() const
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr hiprandRngType type()
    {
        return HIPRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
    }

private:
    hiprandGenerator_t m_generator;

    /// \cond
    template<class T>
    friend class ::hiprand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::hiprand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::hiprand_cpp::normal_distribution;

    template<class T>
    friend class ::hiprand_cpp::lognormal_distribution;

    template<class T>
    friend class ::hiprand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned int DefaultNumDimensions>
constexpr typename scrambled_sobol64_engine<DefaultNumDimensions>::dimensions_num_type
    scrambled_sobol64_engine<DefaultNumDimensions>::default_num_dimensions;
/// \endcond

/// \typedef philox4x32_10;
/// \brief Typedef of hiprand_cpp::philox4x32_10_engine PRNG engine with default seed (#HIPRAND_PHILOX4x32_DEFAULT_SEED).
typedef philox4x32_10_engine<> philox4x32_10;
/// \typedef xorwow
/// \brief Typedef of hiprand_cpp::xorwow_engine PRNG engine with default seed (#HIPRAND_XORWOW_DEFAULT_SEED).
typedef xorwow_engine<> xorwow;
/// \typedef mrg32k3a
/// \brief Typedef of hiprand_cpp::mrg32k3a_engine PRNG engine with default seed (#HIPRAND_MRG32K3A_DEFAULT_SEED).
typedef mrg32k3a_engine<> mrg32k3a;
/// \typedef mtgp32
/// \brief Typedef of hiprand_cpp::mtgp32_engine PRNG engine with default seed (#HIPRAND_MTGP32_DEFAULT_SEED).
typedef mtgp32_engine<> mtgp32;
/// \typedef mt19937
/// \brief Typedef of hiprand_cpp::mt19937_engine PRNG engine with default seed (#HIPRAND_MT19937_DEFAULT_SEED).
typedef mt19937_engine<> mt19937;
/// \typedef sobol32
/// \brief Typedef of hiprand_cpp::sobol32_engine QRNG engine with default number of dimensions (1).
typedef sobol32_engine<> sobol32;
/// \typedef scrambled_sobol32
/// \brief Typedef of hiprand_cpp::scrambled_sobol32_engine QRNG engine with default number of dimensions (1).
typedef scrambled_sobol32_engine<> scrambled_sobol32;
/// \typedef sobol64
/// \brief Typedef of hiprand_cpp::sobol64_engine QRNG engine with default number of dimensions (1).
typedef sobol64_engine<> sobol64;
/// \typedef scrambled_sobol64
/// \brief Typedef of hiprand_cpp::scrambled_sobol64_engine QRNG engine with default number of dimensions (1).
typedef scrambled_sobol64_engine<> scrambled_sobol64;

/// \typedef default_random_engine
/// \brief Default random engine.
typedef xorwow default_random_engine;

/// \typedef random_device
///
/// \brief A non-deterministic uniform random number generator.
///
/// hiprand_cpp::random_device is non-deterministic uniform random number generator,
/// or a pseudo-random number engine if there is no support for non-deterministic
/// random number generation. It's implemented as a typedef of std::random_device.
///
/// For practical use hiprand_cpp::random_device is generally only used to seed a PRNG
/// such as \ref hiprand_cpp::mtgp32_engine.
///
/// Example:
/// \code
/// #include <hiprand/hiprand.hpp>
///
/// int main()
/// {
///     const size_t size = 8192;
///     unsigned int * output;
///     hipMalloc(&output, size * sizeof(unsigned int));
///
///     hiprand_cpp::random_device rd;
///     hiprand_cpp::mtgp32 engine(rd()); // seed engine with a real random value, if available
///     hiprand_cpp::normal_distribution<float> dist(0.0, 1.5);
///     dist(engine, output, size);
/// }
/// \endcode
typedef std::random_device random_device;

/// \brief Returns hipRAND version.
/// \return hipRAND version number as an \p int value.
inline int version()
{
    int x;
    hiprandStatus_t status = hiprandGetVersion(&x);
    if(status != HIPRAND_STATUS_SUCCESS)
    {
        throw hiprand_cpp::error(status);
    }
    return x;
}

/// @} // end of group hiprandhostcpp

} // end namespace hiprand_cpp

#endif // #if __cplusplus >= 201103L
#endif // HIPRAND_HPP_
