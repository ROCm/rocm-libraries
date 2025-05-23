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

#ifndef ROCRAND_HPP_
#define ROCRAND_HPP_

// At least C++11 required
#if defined(__cplusplus) && __cplusplus >= 201103L

    // IWYU pragma: begin_exports
    #include "rocrand/rocrand.h"
    #include "rocrand/rocrand_kernel.h"
    // IWYU pragma: end_exports

    #include <cassert>
    #include <exception>
    #include <limits>
    #include <random>
    #include <sstream>
    #include <string>
    #include <type_traits>

namespace rocrand_cpp
{

/// \rocrand_internal \addtogroup rocrandhostcpp
/// @{

/// \class error
/// \brief A run-time rocRAND error.
///
/// The error class represents an error returned
/// by a rocRAND function.
class error : public std::exception
{
public:
    /// rocRAND error code type
    typedef rocrand_status error_type;

    /// Constructs new error object from error code \p error.
    ///
    /// \param error error code
    explicit error(error_type error) noexcept
        : m_error(error),
          m_error_string(to_string(error))
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
    const char* what() const noexcept override
    {
        return m_error_string.c_str();
    }

    /// Static function which converts the numeric rocRAND
    /// error code \p error to a human-readable string.
    ///
    /// If the error code is unknown, a string containing
    /// "Unknown rocRAND error" along with the error code
    /// \p error will be returned.
    static std::string to_string(error_type error)
    {
        switch(error)
        {
            case ROCRAND_STATUS_SUCCESS:
                return "Success";
            case ROCRAND_STATUS_VERSION_MISMATCH:
                return "Header file and linked library version do not match";
            case ROCRAND_STATUS_NOT_CREATED:
                return "Generator was not created using rocrand_create_generator";
            case ROCRAND_STATUS_ALLOCATION_FAILED:
                return "Memory allocation failed during execution";
            case ROCRAND_STATUS_TYPE_ERROR:
                return "Generator type is wrong";
            case ROCRAND_STATUS_OUT_OF_RANGE:
                return "Argument out of range";
            case ROCRAND_STATUS_LENGTH_NOT_MULTIPLE:
                return "Length requested is not a multiple of dimension";
            case ROCRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
                return "GPU does not have double precision";
            case ROCRAND_STATUS_LAUNCH_FAILURE:
                return "Kernel launch failure";
            case ROCRAND_STATUS_INTERNAL_ERROR:
                return "Internal library error";
            default: {
                std::stringstream s;
                s << "Unknown rocRAND error (" << error << ")";
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
/// \tparam IntType type of generated values. Only \p unsigned \p char, \p unsigned \p short and \p unsigned \p int and \p unsigned \p long \p long \p int type is supported.
template<class IntType = unsigned int>
class uniform_int_distribution
{
    static_assert(std::is_same<unsigned char, IntType>::value
                      || std::is_same<unsigned short, IntType>::value
                      || std::is_same<unsigned long long int, IntType>::value
                      || std::is_same<unsigned int, IntType>::value,
                  "Only unsigned char, unsigned short, unsigned int and unsigned long long int "
                  "types are supported in uniform_int_distribution");

public:
    /// See description for IntType template parameter.
    typedef IntType result_type;

    /// Default constructor
    uniform_int_distribution()
    {
    }

    /// Resets distribution's internal state if there is any.
    static void reset()
    {
    }

    /// Returns the smallest possible value that can be generated.
    static constexpr IntType min()
    {
        return 0;
    }

    /// Returns the largest possible value that can be generated.
    static constexpr IntType max()
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
    /// * If generator \p g is a quasi-random number generator (`rocrand_cpp::sobol32_engine`),
    /// then \p size must be a multiple of that generator's dimension.
    ///
    /// See also: rocrand_generate(), rocrand_generate_char(), rocrand_generate_short()
    template<class Generator>
    void operator()(Generator& g, IntType * output, size_t size)
    {
        rocrand_status status;
        status = this->generate(g, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// Returns \c true if the distribution is the same as \p other.
    bool operator==(const uniform_int_distribution<IntType>& other) const
    {
        (void) other;
        return true;
    }

    /// Returns \c true if the distribution is different from \p other.
    bool operator!=(const uniform_int_distribution<IntType>& other) const
    {
        return !(*this == other);
    }

private:
    template<class Generator>
    static rocrand_status generate(Generator& g, unsigned char * output, size_t size)
    {
        return rocrand_generate_char(g.m_generator, output, size);
    }

    template<class Generator>
    static rocrand_status generate(Generator& g, unsigned short * output, size_t size)
    {
        return rocrand_generate_short(g.m_generator, output, size);
    }

    template<class Generator>
    static rocrand_status generate(Generator& g, unsigned int * output, size_t size)
    {
        return rocrand_generate(g.m_generator, output, size);
    }

    template<class Generator>
    static rocrand_status generate(Generator& g, unsigned long long int* output, size_t size)
    {
        return rocrand_generate_long_long(g.m_generator, output, size);
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
    /// See description for RealType template parameter.
    typedef RealType result_type;

    /// Default constructor
    uniform_real_distribution()
    {
    }

    /// Resets distribution's internal state if there is any.
    static void reset()
    {
    }

    /// Returns the smallest possible value that can be generated.
    static constexpr RealType min()
    {
        return static_cast<RealType>(ROCRAND_2POW32_INV_DOUBLE);
    }

    /// Returns the largest possible value that can be generated.
    static constexpr RealType max()
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
    /// * If generator \p g is a quasi-random number generator (`rocrand_cpp::sobol32_engine`),
    /// then \p size must be a multiple of that generator's dimension.
    ///
    /// See also: rocrand_generate_uniform(), rocrand_generate_uniform_double(), rocrand_generate_uniform_half()
    template<class Generator>
    void operator()(Generator& g, RealType * output, size_t size)
    {
        rocrand_status status;
        status = this->generate(g, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// Returns \c true if the distribution is the same as \p other.
    bool operator==(const uniform_real_distribution<RealType>& other) const
    {
        (void) other;
        return true;
    }

    /// Returns \c true if the distribution is different from \p other.
    bool operator!=(const uniform_real_distribution<RealType>& other) const
    {
        return !(*this == other);
    }

private:
    template<class Generator>
    static rocrand_status generate(Generator& g, float * output, size_t size)
    {
        return rocrand_generate_uniform(g.m_generator, output, size);
    }

    template<class Generator>
    static rocrand_status generate(Generator& g, double * output, size_t size)
    {
        return rocrand_generate_uniform_double(g.m_generator, output, size);
    }

    template<class Generator>
    static rocrand_status generate(Generator& g, half * output, size_t size)
    {
        return rocrand_generate_uniform_half(g.m_generator, output, size);
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
    /// See description for RealType template parameter.
    typedef RealType result_type;

    /// \class param_type
    /// \brief The type of the distribution parameter set.
    class param_type
    {
    public:
        /// Alias for convenience
        using distribution_type = normal_distribution<RealType>;

        /// \brief Constructs a \p param_type object with the
        /// given distribution parameters.
        /// \param mean mean
        /// \param stddev standard deviation
        param_type(RealType mean = 0.0, RealType stddev = 1.0)
            : m_mean(mean), m_stddev(stddev)
        {
        }

        /// Copy constructor
        param_type(const param_type& params) = default;

        /// Copy assignment operator
        param_type& operator=(const param_type& params) = default;

        /// \brief Returns the deviation distribution parameter.
        ///
        /// The default value is 0.0.
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
        bool operator==(const param_type& other) const
        {
            return m_mean == other.m_mean && m_stddev == other.m_stddev;
        }

        /// Returns \c true if the param_type is different from \p other.
        bool operator!=(const param_type& other) const
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
    explicit normal_distribution(const param_type& params)
        : m_params(params)
    {
    }

    /// Resets distribution's internal state if there is any.
    static void reset()
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
    static constexpr RealType min()
    {
        return std::numeric_limits<RealType>::lowest();
    }

    /// Returns the largest possible value that can be generated.
    static constexpr RealType max()
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
    /// * If generator \p g is a quasi-random number generator (`rocrand_cpp::sobol32_engine`),
    /// then \p size must be a multiple of that generator's dimension.
    ///
    /// See also: rocrand_generate_normal(), rocrand_generate_normal_double(), rocrand_generate_normal_half()
    template<class Generator>
    void operator()(Generator& g, RealType * output, size_t size)
    {
        rocrand_status status;
        status = this->generate(g, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \brief Returns \c true if the distribution is the same as \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator==(const normal_distribution<RealType>& other) const 
    {
        return this->m_params == other.m_params;
    }

    /// \brief Returns \c true if the distribution is different from \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator!=(const normal_distribution<RealType>& other) const
    {
        return !(*this == other);
    }

private:
    template<class Generator>
    rocrand_status generate(Generator& g, float * output, size_t size)
    {
        return rocrand_generate_normal(
            g.m_generator, output, size, this->mean(), this->stddev()
        );
    }

    template<class Generator>
    rocrand_status generate(Generator& g, double * output, size_t size)
    {
        return rocrand_generate_normal_double(
            g.m_generator, output, size, this->mean(), this->stddev()
        );
    }

    template<class Generator>
    rocrand_status generate(Generator& g, half * output, size_t size)
    {
        return rocrand_generate_normal_half(
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
    /// See description for RealType template parameter.
    typedef RealType result_type;

    /// \class param_type
    /// \brief The type of the distribution parameter set.
    class param_type
    {
    public:
        /// Alias for convenience
        using distribution_type = lognormal_distribution<RealType>;

        /// \brief Constructs a \p param_type object with the
        /// given distribution parameters.
        /// \param m mean
        /// \param s standard deviation
        param_type(RealType m = 0.0, RealType s = 1.0)
            : m_mean(m), m_stddev(s)
        {
        }

        /// Copy constructor
        param_type(const param_type& params) = default;

        /// Copy assignment operator
        param_type& operator=(const param_type& params) = default;

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
        bool operator==(const param_type& other) const
        {
            return m_mean == other.m_mean && m_stddev == other.m_stddev;
        }

        /// Returns \c true if the param_type is different from \p other.
        bool operator!=(const param_type& other) const
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
    explicit lognormal_distribution(const param_type& params)
        : m_params(params)
    {
    }

    /// Resets distribution's internal state if there is any.
    static void reset()
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
    static constexpr RealType min()
    {
        return 0;
    }

    /// Returns the largest possible value that can be generated.
    static RealType max()
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
    /// * If generator \p g is a quasi-random number generator (`rocrand_cpp::sobol32_engine`),
    /// then \p size must be a multiple of that generator's dimension.
    ///
    /// See also: rocrand_generate_log_normal(), rocrand_generate_log_normal_double()
    template<class Generator>
    void operator()(Generator& g, RealType * output, size_t size)
    {
        rocrand_status status;
        status = this->generate(g, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \brief Returns \c true if the distribution is the same as \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator==(const lognormal_distribution<RealType>& other) const
    {
        return this->m_params == other.m_params;
    }

    /// \brief Returns \c true if the distribution is different from \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator!=(const lognormal_distribution<RealType>& other) const
    {
        return !(*this == other);
    }

private:
    template<class Generator>
    rocrand_status generate(Generator& g, float * output, size_t size)
    {
        return rocrand_generate_log_normal(
            g.m_generator, output, size, this->m(), this->s()
        );
    }

    template<class Generator>
    rocrand_status generate(Generator& g, double * output, size_t size)
    {
        return rocrand_generate_log_normal_double(
            g.m_generator, output, size, this->m(), this->s()
        );
    }

    template<class Generator>
    rocrand_status generate(Generator& g, half * output, size_t size)
    {
        return rocrand_generate_log_normal_half(
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
    /// See description for IntType template parameter.
    typedef IntType result_type;

    /// \class param_type
    /// \brief The type of the distribution parameter set.
    class param_type
    {
    public:
        /// Alias for convenience.
        using distribution_type = poisson_distribution<IntType>;

        /// \brief Constructs a \p param_type object with the
        /// given mean.
        /// \param mean mean to use for the distribution
        param_type(double mean = 1.0)
            : m_mean(mean)
        {
        }

        /// Copy constructor
        param_type(const param_type& params) = default;

        /// Copy assignment operator
        param_type& operator=(const param_type& params) = default;

        /// \brief Returns the mean distribution parameter.
        ///
        /// The mean (also known as lambda) is the average number
        /// of events per interval. The default value is 1.0.
        double mean() const
        {
            return m_mean;
        }

        /// Returns \c true if the param_type is the same as \p other.
        bool operator==(const param_type& other) const
        {
            return m_mean == other.m_mean;
        }

        /// Returns \c true if the param_type is different from \p other.
        bool operator!=(const param_type& other) const
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
    explicit poisson_distribution(const param_type& params)
        : m_params(params)
    {
    }

    /// Resets distribution's internal state if there is any.
    static void reset()
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
    static constexpr IntType min()
    {
        return 0;
    }

    /// Returns the largest possible value that can be generated.
    static constexpr IntType max()
    {
        return std::numeric_limits<IntType>::max();
    }

    /// Returns the distribution parameter object
    param_type param()
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
    /// * If generator \p g is a quasi-random number generator (`rocrand_cpp::sobol32_engine`),
    /// then \p size must be a multiple of that generator's dimension.
    ///
    /// See also: rocrand_generate_poisson()
    template<class Generator>
    void operator()(Generator& g, IntType * output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate_poisson(g.m_generator, output, size, this->mean());
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \brief Returns \c true if the distribution is the same as \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator==(const poisson_distribution<IntType>& other) const
    {
        return this->m_params == other.m_params;
    }

    /// \brief Returns \c true if the distribution is different from \p other.
    ///
    /// Two distribution are equal, if their parameters are equal.
    bool operator!=(const poisson_distribution<IntType>& other) const
    {
        return !(*this == other);
    }

private:
    param_type m_params;
};

/// \brief Pseudorandom number engine based Philox algorithm.
///
/// It generates random numbers of type \p unsigned \p int on the interval [0; 2^32 - 1].
/// Random numbers are generated in sets of four.
template<unsigned long long DefaultSeed = ROCRAND_PHILOX4x32_DEFAULT_SEED>
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
    typedef rocrand_ordering order_type;
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
    /// \param order_value ordering of the sequences generated by the engine, see also order()
    ///
    /// See also: rocrand_create_generator()
    philox4x32_10_engine(seed_type   seed_value   = DefaultSeed,
                         offset_type offset_value = 0,
                         order_type  order_value  = ROCRAND_ORDERING_PSEUDO_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
        try
        {
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->order(order_value);
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
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
    /// \param generator rocRAND generator
    explicit philox4x32_10_engine(rocrand_generator& generator)
        : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    philox4x32_10_engine(const philox4x32_10_engine&) = delete;

    philox4x32_10_engine& operator=(const philox4x32_10_engine&) = delete;

    /// \brief Move construct from an other engine, moving the state over.
    ///
    /// \param rhs the engine to move-from
    ///
    /// - The moved-from engine is safe to assign to or destroy, but otherwise cannot be used.
    /// - This engine will continue the sequence generated by `rhs`.
    philox4x32_10_engine(philox4x32_10_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \brief Move assign from an other engine, moving the state over.
    ///
    /// \param rhs the engine to move-from
    ///
    /// - The moved-from engine is safe to assign to or destroy, but otherwise cannot be used.
    /// - This engine will continue the sequence generated by `rhs`.
    philox4x32_10_engine& operator=(philox4x32_10_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// Destructs the engine.
    ///
    /// See also: rocrand_destroy_generator()
    ~philox4x32_10_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \brief Sets the random number engine's \p hipStream for kernel launches.
    /// \param value new \p hipStream to use
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
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
    /// See also: rocrand_set_ordering()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
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
    /// See also: rocrand_set_offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \brief Sets the seed of the pseudo-random number engine.
    ///
    /// The seed is used to construct the initial state of an engine.
    ///
    /// - This operation resets the engine's internal state.
    /// - This operation does not change the engine's offset.
    ///
    /// \param value New seed value
    ///
    /// See also: rocrand_set_seed()
    void seed(seed_type value)
    {
        rocrand_status status = rocrand_set_seed(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
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
    /// See also: rocrand_generate()
    template<class Generator>
    void operator()(result_type * output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// Returns the smallest possible value that can be generated by the engine.
    static constexpr result_type min()
    {
        return 0;
    }

    /// Returns the largest possible value that can be generated by the engine.
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// Returns type of the rocRAND pseudo-random number generator associated with the engine.
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_PHILOX4_32_10;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename philox4x32_10_engine<DefaultSeed>::seed_type philox4x32_10_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Pseudorandom number engine based XORWOW algorithm.
///
/// xorwow_engine is a xorshift pseudorandom
/// number engine based on XORWOW algorithm. It produces random numbers
/// of type \p unsigned \p int on the interval [0; 2^32 - 1].
template<unsigned long long DefaultSeed = ROCRAND_XORWOW_DEFAULT_SEED>
class xorwow_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(seed_type,offset_type,order_type)
    xorwow_engine(seed_type   seed_value   = DefaultSeed,
                  offset_type offset_value = 0,
                  order_type  order_value  = ROCRAND_ORDERING_PSEUDO_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
        try
        {
            this->order(order_value);
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit xorwow_engine(rocrand_generator& generator)
        : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    xorwow_engine(const xorwow_engine&) = delete;

    xorwow_engine& operator=(const xorwow_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    xorwow_engine(xorwow_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    xorwow_engine& operator=(xorwow_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~xorwow_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        rocrand_status status = rocrand_set_seed(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type * output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_XORWOW;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename xorwow_engine<DefaultSeed>::seed_type xorwow_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Pseudorandom number engine based MRG31k3p CMRG.
///
/// mrg31k3p_engine is an implementation of MRG31k3p pseudorandom number generator,
/// which is a Combined Multiple Recursive Generator (CMRG) created by Pierre L'Ecuyer.
/// It produces random 32-bit \p unsigned \p int values on the interval [0; 2^32 - 1].
template<unsigned long long DefaultSeed = ROCRAND_MRG31K3P_DEFAULT_SEED>
class mrg31k3p_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(seed_type,offset_type,order_type)
    mrg31k3p_engine(seed_type   seed_value   = DefaultSeed,
                    offset_type offset_value = 0,
                    order_type  order_value  = ROCRAND_ORDERING_PSEUDO_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
        try
        {
            this->order(order_value);
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit mrg31k3p_engine(rocrand_generator& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    mrg31k3p_engine(const mrg31k3p_engine&) = delete;

    mrg31k3p_engine& operator=(const mrg31k3p_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    mrg31k3p_engine(mrg31k3p_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    mrg31k3p_engine& operator=(mrg31k3p_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~mrg31k3p_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        rocrand_status status = rocrand_set_seed(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 1;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_MRG31K3P;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr
    typename mrg31k3p_engine<DefaultSeed>::seed_type mrg31k3p_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Pseudorandom number engine based MRG32k3a CMRG.
///
/// mrg32k3a_engine is an implementation of MRG32k3a pseudorandom number generator,
/// which is a Combined Multiple Recursive Generator (CMRG) created by Pierre L'Ecuyer.
/// It produces random 32-bit \p unsigned \p int values on the interval [0; 2^32 - 1].
template<unsigned long long DefaultSeed = ROCRAND_MRG32K3A_DEFAULT_SEED>
class mrg32k3a_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(seed_type,offset_type,order_type)
    mrg32k3a_engine(seed_type   seed_value   = DefaultSeed,
                    offset_type offset_value = 0,
                    order_type  order_value  = ROCRAND_ORDERING_PSEUDO_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
        try
        {
            this->order(order_value);
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit mrg32k3a_engine(rocrand_generator& generator)
        : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    mrg32k3a_engine(const mrg32k3a_engine&) = delete;

    mrg32k3a_engine& operator=(const mrg32k3a_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    mrg32k3a_engine(mrg32k3a_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    mrg32k3a_engine& operator=(mrg32k3a_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~mrg32k3a_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        rocrand_status status = rocrand_set_seed(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type * output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 1;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_MRG32K3A;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename mrg32k3a_engine<DefaultSeed>::seed_type mrg32k3a_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Random number engine based on the Mersenne Twister for Graphic Processors algorithm.
///
/// mtgp32_engine is a random number engine based on the Mersenne Twister
/// for Graphic Processors algorithm, which is a version of well-known
/// Mersenne Twister algorithm. It produces high quality random numbers of type \p unsigned \p int
/// on the interval [0; 2^32 - 1].
template<unsigned long long DefaultSeed = 0>
class mtgp32_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \brief Constructs the pseudo-random number engine.
    ///
    /// MTGP32 engine does not accept offset.
    ///
    /// \param seed_value seed value to use in the initialization of the internal state, see also seed()
    /// \param order_value ordering value from the rocrand_ordering enum
    ///
    /// See also: rocrand_create_generator()
    mtgp32_engine(seed_type  seed_value  = DefaultSeed,
                  order_type order_value = ROCRAND_ORDERING_PSEUDO_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
        try
        {
            this->order(order_value);
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit mtgp32_engine(rocrand_generator& generator)
        : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    mtgp32_engine(const mtgp32_engine&) = delete;

    mtgp32_engine& operator=(const mtgp32_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    mtgp32_engine(mtgp32_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    mtgp32_engine& operator=(mtgp32_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~mtgp32_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        rocrand_status status = rocrand_set_seed(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type * output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_MTGP32;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename mtgp32_engine<DefaultSeed>::seed_type mtgp32_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Random number engine based on the LFSR113 algorithm.
///
/// lfsr113_engine is an implementation of LFSR113 pseudorandom number generator,
/// which is a linear feedback shift resgisters (LFSR) based generator created by Pierre L'Ecuyer.
/// It produces random 32-bit \p unsigned \p int values on the interval [0; 2^32 - 1].
template<unsigned int DefaultSeedX = ROCRAND_LFSR113_DEFAULT_SEED_X,
         unsigned int DefaultSeedY = ROCRAND_LFSR113_DEFAULT_SEED_Y,
         unsigned int DefaultSeedZ = ROCRAND_LFSR113_DEFAULT_SEED_Z,
         unsigned int DefaultSeedW = ROCRAND_LFSR113_DEFAULT_SEED_W>
class lfsr113_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef uint4 seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed
        = {DefaultSeedX, DefaultSeedY, DefaultSeedZ, DefaultSeedW};

    /// \brief Constructs the pseudo-random number engine.
    ///
    /// LFSR113 does not accept offset.
    ///
    /// \param seed_value seed value to use in the initialization of the internal state, see also seed()
    /// \param order_value ordering of the sequences generated by the engine, see also order()
    ///
    /// See also: rocrand_create_generator()
    lfsr113_engine(seed_type  seed_value = {DefaultSeedX, DefaultSeedY, DefaultSeedZ, DefaultSeedW},
                   order_type order_value = ROCRAND_ORDERING_PSEUDO_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
        try
        {
            this->order(order_value);
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc lfsr113_engine::lfsr113_engine(seed_type,order_type)
    lfsr113_engine(unsigned long long seed_value,
                   order_type         order_value = ROCRAND_ORDERING_PSEUDO_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
        try
        {
            this->order(order_value);
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit lfsr113_engine(rocrand_generator& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    lfsr113_engine(const lfsr113_engine&) = delete;

    lfsr113_engine& operator=(const lfsr113_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    lfsr113_engine(lfsr113_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    lfsr113_engine& operator=(lfsr113_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~lfsr113_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(unsigned long long value)
    {
        rocrand_status status = rocrand_set_seed(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        rocrand_status status = rocrand_set_seed_uint4(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_LFSR113;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned int DefaultSeedX,
         unsigned int DefaultSeedY,
         unsigned int DefaultSeedZ,
         unsigned int DefaultSeedW>
constexpr typename lfsr113_engine<DefaultSeedX, DefaultSeedY, DefaultSeedZ, DefaultSeedW>::seed_type
    lfsr113_engine<DefaultSeedX, DefaultSeedY, DefaultSeedZ, DefaultSeedW>::default_seed;
/// \endcond

/// \brief Random number engine based on the Mersenne Twister algorithm.
///
/// mt19937 is a random number engine based on the Mersenne Twister algorithm
/// as proposed in "Mersenne Twister: A 623-Dimensionally Equidistributed Uniform
/// Pseudo-Random Number Generator". It produces high quality random numbers of
/// type \p unsigned \p int on the interval [0; 2^32 - 1].
template<unsigned long long DefaultSeed = 0ULL>
class mt19937_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \brief Constructs the pseudo-random number engine.
    ///
    /// MT19937 does not accept offset.
    ///
    /// \param seed_value seed value to use in the initialization of the internal state, see also seed()
    /// \param order_value ordering of the sequences generated by the engine, see also order()
    ///
    /// See also: rocrand_create_generator()
    mt19937_engine(seed_type  seed_value  = DefaultSeed,
                   order_type order_value = ROCRAND_ORDERING_PSEUDO_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
        try
        {
            this->order(order_value);
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit mt19937_engine(rocrand_generator& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    mt19937_engine(const mt19937_engine&) = delete;

    mt19937_engine& operator=(const mt19937_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    mt19937_engine(mt19937_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    mt19937_engine& operator=(mt19937_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~mt19937_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        rocrand_status status = rocrand_set_seed(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_MT19937;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
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
/// The engine produces random unsigned integers on the interval [0, 2^32 - 1].
template<unsigned int DefaultNumDimensions = 1>
class sobol32_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
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
    /// \param order_value ordering of the sequences generated by the engine, see also order()
    ///
    /// See also: rocrand_create_generator()
    sobol32_engine(dimensions_num_type num_of_dimensions = DefaultNumDimensions,
                   offset_type         offset_value      = 0,
                   order_type          order_value       = ROCRAND_ORDERING_QUASI_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
        try
        {
            this->order(order_value);
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->dimensions(num_of_dimensions);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit sobol32_engine(rocrand_generator& generator)
        : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    sobol32_engine(const sobol32_engine&) = delete;

    sobol32_engine& operator=(const sobol32_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    sobol32_engine(sobol32_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    sobol32_engine& operator=(sobol32_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~sobol32_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
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
    /// See also: rocrand_set_quasi_random_generator_dimensions()
    void dimensions(dimensions_num_type value)
    {
        rocrand_status status =
            rocrand_set_quasi_random_generator_dimensions(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
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
    /// See also: rocrand_generate()
    template<class Generator>
    void operator()(result_type * output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_QUASI_SOBOL32;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned int DefaultNumDimensions>
constexpr typename sobol32_engine<DefaultNumDimensions>::dimensions_num_type
    sobol32_engine<DefaultNumDimensions>::default_num_dimensions;
/// \endcond

/// \brief Sobol's scrambled quasi-random sequence generator
///
/// scrambled_sobol32_engine is a quasi-random number engine which produces scrambled Sobol sequences.
/// This implementation supports generating sequences in up to 20,000 dimensions.
/// The engine produces random unsigned integers on the interval [0, 2^32 - 1].
template<unsigned int DefaultNumDimensions = 1>
class scrambled_sobol32_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
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
    /// \param order_value ordering value from the rocrand_ordering enum
    ///
    /// See also: rocrand_create_generator()
    scrambled_sobol32_engine(dimensions_num_type num_of_dimensions = DefaultNumDimensions,
                             offset_type         offset_value      = 0,
                             order_type          order_value       = ROCRAND_ORDERING_QUASI_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
        try
        {
            this->order(order_value);
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->dimensions(num_of_dimensions);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit scrambled_sobol32_engine(rocrand_generator& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    scrambled_sobol32_engine(const scrambled_sobol32_engine&) = delete;

    scrambled_sobol32_engine& operator=(const scrambled_sobol32_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    scrambled_sobol32_engine(scrambled_sobol32_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    scrambled_sobol32_engine& operator=(scrambled_sobol32_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~scrambled_sobol32_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
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
    /// See also: rocrand_set_quasi_random_generator_dimensions()
    void dimensions(dimensions_num_type value)
    {
        rocrand_status status
            = rocrand_set_quasi_random_generator_dimensions(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
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
    /// See also: rocrand_generate()
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL32;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned int DefaultNumDimensions>
constexpr typename scrambled_sobol32_engine<DefaultNumDimensions>::dimensions_num_type
    scrambled_sobol32_engine<DefaultNumDimensions>::default_num_dimensions;
/// \endcond

/// \brief Sobol's quasi-random sequence generator
///
/// sobol64 is a quasi-random number engine which produces Sobol sequences.
/// This implementation supports generating sequences in up to 20,000 dimensions.
/// The engine produces random unsigned integers on the interval [0, 2^64 - 1].
template<unsigned int DefaultNumDimensions = 1>
class sobol64_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned long long int result_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long int offset_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
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
    /// \param order_value ordering of the sequences generated by the engine, see also order()
    ///
    /// See also: rocrand_create_generator()
    sobol64_engine(dimensions_num_type num_of_dimensions = DefaultNumDimensions,
                   offset_type         offset_value      = 0,
                   order_type          order_value       = ROCRAND_ORDERING_QUASI_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
        try
        {
            this->order(order_value);
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->dimensions(num_of_dimensions);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit sobol64_engine(rocrand_generator& generator)
        : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    sobol64_engine(const sobol64_engine&) = delete;

    sobol64_engine& operator=(const sobol64_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    sobol64_engine(sobol64_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    sobol64_engine& operator=(sobol64_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~sobol64_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
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
    /// See also: rocrand_set_quasi_random_generator_dimensions()
    void dimensions(dimensions_num_type value)
    {
        rocrand_status status =
            rocrand_set_quasi_random_generator_dimensions(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
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
    /// See also: rocrand_generate()
    template<class Generator>
    void operator()(result_type * output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate_long_long(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS) throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<result_type>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_QUASI_SOBOL64;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned int DefaultNumDimensions>
constexpr typename sobol64_engine<DefaultNumDimensions>::dimensions_num_type
    sobol64_engine<DefaultNumDimensions>::default_num_dimensions;
/// \endcond

/// \brief Sobol's scrambled quasi-random sequence generator
///
/// scrambled_sobol64_engine is a quasi-random number engine which produces scrambled Sobol sequences.
/// This implementation supports generating sequences in up to 20,000 dimensions.
/// The engine produces random unsigned long long integers on the interval [0, 2^64 - 1].
template<unsigned int DefaultNumDimensions = 1>
class scrambled_sobol64_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned long long int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long int offset_type;
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
    /// \param order_value ordering of the sequences generated by the engine, see also order()
    ///
    /// See also: rocrand_create_generator()
    scrambled_sobol64_engine(dimensions_num_type num_of_dimensions = DefaultNumDimensions,
                             offset_type         offset_value      = 0,
                             order_type          order_value       = ROCRAND_ORDERING_QUASI_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
        try
        {
            this->order(order_value);
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->dimensions(num_of_dimensions);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit scrambled_sobol64_engine(rocrand_generator& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    scrambled_sobol64_engine(const scrambled_sobol64_engine&) = delete;

    scrambled_sobol64_engine& operator=(const scrambled_sobol64_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    scrambled_sobol64_engine(scrambled_sobol64_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    scrambled_sobol64_engine& operator=(scrambled_sobol64_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~scrambled_sobol64_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
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
    /// See also: rocrand_set_quasi_random_generator_dimensions()
    void dimensions(dimensions_num_type value)
    {
        rocrand_status status
            = rocrand_set_quasi_random_generator_dimensions(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
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
    /// See also: rocrand_generate()
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate_long_long(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<result_type>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_QUASI_SCRAMBLED_SOBOL64;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned int DefaultNumDimensions>
constexpr typename scrambled_sobol64_engine<DefaultNumDimensions>::dimensions_num_type
    scrambled_sobol64_engine<DefaultNumDimensions>::default_num_dimensions;
/// \endcond

/// \brief Pseudorandom number engine based on 2 state ThreeFry.
///
/// It generates random numbers of type \p unsigned \p int on the interval [0; 2^32 - 1].
/// Random numbers are generated in sets of two.
template<unsigned long long DefaultSeed = 0>
class threefry2x32_20_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(seed_type, offset_type, order_type)
    threefry2x32_20_engine(seed_type   seed_value   = DefaultSeed,
                           offset_type offset_value = 0,
                           order_type  order_value  = ROCRAND_ORDERING_PSEUDO_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
        try
        {
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->order(order_value);
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit threefry2x32_20_engine(rocrand_generator& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    threefry2x32_20_engine(const threefry2x32_20_engine&) = delete;

    threefry2x32_20_engine& operator=(const threefry2x32_20_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    threefry2x32_20_engine(threefry2x32_20_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    threefry2x32_20_engine& operator=(threefry2x32_20_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~threefry2x32_20_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        rocrand_status status = rocrand_set_seed(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_THREEFRY2_32_20;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename threefry2x32_20_engine<DefaultSeed>::seed_type
    threefry2x32_20_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Pseudorandom number engine based 2 state ThreeFry.
///
/// It generates random numbers of type \p unsigned \p int on the interval [0; 2^62 - 1].
/// Random numbers are generated in sets of two.
template<unsigned long long DefaultSeed = 0>
class threefry2x64_20_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned long long result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(seed_type, offset_type, order_type)
    threefry2x64_20_engine(seed_type   seed_value   = DefaultSeed,
                           offset_type offset_value = 0,
                           order_type  order_value  = ROCRAND_ORDERING_PSEUDO_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
        try
        {
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->order(order_value);
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit threefry2x64_20_engine(rocrand_generator& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    threefry2x64_20_engine(const threefry2x64_20_engine&) = delete;

    threefry2x64_20_engine& operator=(const threefry2x64_20_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    threefry2x64_20_engine(threefry2x64_20_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    threefry2x64_20_engine& operator=(threefry2x64_20_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~threefry2x64_20_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        rocrand_status status = rocrand_set_seed(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate_long_long(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_THREEFRY2_64_20;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename threefry2x64_20_engine<DefaultSeed>::seed_type
    threefry2x64_20_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Pseudorandom number engine based on 2 state ThreeFry.
///
/// It generates random numbers of type \p unsigned \p int on the interval [0; 2^32 - 1].
/// Random numbers are generated in sets of two.
template<unsigned long long DefaultSeed = 0>
class threefry4x32_20_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned int result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(seed_type, offset_type, order_type)
    threefry4x32_20_engine(seed_type   seed_value   = DefaultSeed,
                           offset_type offset_value = 0,
                           order_type  order_value  = ROCRAND_ORDERING_PSEUDO_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
        try
        {
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->order(order_value);
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit threefry4x32_20_engine(rocrand_generator& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    threefry4x32_20_engine(const threefry4x32_20_engine&) = delete;

    threefry4x32_20_engine& operator=(const threefry4x32_20_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    threefry4x32_20_engine(threefry4x32_20_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    threefry4x32_20_engine& operator=(threefry4x32_20_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~threefry4x32_20_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        rocrand_status status = rocrand_set_seed(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_THREEFRY4_32_20;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename threefry4x32_20_engine<DefaultSeed>::seed_type
    threefry4x32_20_engine<DefaultSeed>::default_seed;
/// \endcond

/// \brief Pseudorandom number engine based 2 state ThreeFry.
///
/// It generates random numbers of type \p unsigned \p int on the interval [0; 2^62 - 1].
/// Random numbers are generated in sets of two.
template<unsigned long long DefaultSeed = 0>
class threefry4x64_20_engine
{
public:
    /// \copydoc philox4x32_10_engine::result_type
    typedef unsigned long long result_type;
    /// \copydoc philox4x32_10_engine::order_type
    typedef rocrand_ordering order_type;
    /// \copydoc philox4x32_10_engine::offset_type
    typedef unsigned long long offset_type;
    /// \copydoc philox4x32_10_engine::seed_type
    typedef unsigned long long seed_type;
    /// \copydoc philox4x32_10_engine::default_seed
    static constexpr seed_type default_seed = DefaultSeed;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(seed_type, offset_type, order_type)
    threefry4x64_20_engine(seed_type   seed_value   = DefaultSeed,
                           offset_type offset_value = 0,
                           order_type  order_value  = ROCRAND_ORDERING_PSEUDO_DEFAULT)
    {
        rocrand_status status;
        status = rocrand_create_generator(&m_generator, this->type());
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
        try
        {
            if(offset_value > 0)
            {
                this->offset(offset_value);
            }
            this->order(order_value);
            this->seed(seed_value);
        }
        catch(...)
        {
            (void)rocrand_destroy_generator(m_generator);
            throw;
        }
    }

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(rocrand_generator&)
    explicit threefry4x64_20_engine(rocrand_generator& generator) : m_generator(generator)
    {
        if(generator == NULL)
        {
            throw rocrand_cpp::error(ROCRAND_STATUS_NOT_CREATED);
        }
        generator = NULL;
    }

    threefry4x64_20_engine(const threefry4x64_20_engine&) = delete;

    threefry4x64_20_engine& operator=(const threefry4x64_20_engine&) = delete;

    /// \copydoc philox4x32_10_engine::philox4x32_10_engine(philox4x32_10_engine&&)
    threefry4x64_20_engine(threefry4x64_20_engine&& rhs) noexcept : m_generator(rhs.m_generator)
    {
        rhs.m_generator = nullptr;
    }

    /// \copydoc philox4x32_10_engine::operator=(philox4x32_10_engine&&)
    threefry4x64_20_engine& operator=(threefry4x64_20_engine&& rhs) noexcept
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        assert(status == ROCRAND_STATUS_SUCCESS || status == ROCRAND_STATUS_NOT_CREATED);
        (void)status;

        m_generator     = rhs.m_generator;
        rhs.m_generator = nullptr;
        return *this;
    }

    /// \copydoc philox4x32_10_engine::~philox4x32_10_engine()
    ~threefry4x64_20_engine() noexcept(false)
    {
        rocrand_status status = rocrand_destroy_generator(m_generator);
        if(status != ROCRAND_STATUS_SUCCESS && status != ROCRAND_STATUS_NOT_CREATED)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::stream()
    void stream(hipStream_t value)
    {
        rocrand_status status = rocrand_set_stream(m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::order()
    void order(order_type value)
    {
        rocrand_status status = rocrand_set_ordering(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::offset()
    void offset(offset_type value)
    {
        rocrand_status status = rocrand_set_offset(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::seed()
    void seed(seed_type value)
    {
        rocrand_status status = rocrand_set_seed(this->m_generator, value);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::operator()()
    template<class Generator>
    void operator()(result_type* output, size_t size)
    {
        rocrand_status status;
        status = rocrand_generate_long_long(m_generator, output, size);
        if(status != ROCRAND_STATUS_SUCCESS)
            throw rocrand_cpp::error(status);
    }

    /// \copydoc philox4x32_10_engine::min()
    static constexpr result_type min()
    {
        return 0;
    }

    /// \copydoc philox4x32_10_engine::max()
    static constexpr result_type max()
    {
        return std::numeric_limits<unsigned int>::max();
    }

    /// \copydoc philox4x32_10_engine::type()
    static constexpr rocrand_rng_type type()
    {
        return ROCRAND_RNG_PSEUDO_THREEFRY4_64_20;
    }

private:
    rocrand_generator m_generator;

    /// \cond
    template<class T>
    friend class ::rocrand_cpp::uniform_int_distribution;

    template<class T>
    friend class ::rocrand_cpp::uniform_real_distribution;

    template<class T>
    friend class ::rocrand_cpp::normal_distribution;

    template<class T>
    friend class ::rocrand_cpp::lognormal_distribution;

    template<class T>
    friend class ::rocrand_cpp::poisson_distribution;
    /// \endcond
};

/// \cond
template<unsigned long long DefaultSeed>
constexpr typename threefry4x64_20_engine<DefaultSeed>::seed_type
    threefry4x64_20_engine<DefaultSeed>::default_seed;
/// \endcond

/// \typedef philox4x32_10;
/// \brief Typedef of rocrand_cpp::philox4x32_10_engine PRNG engine with default seed (#ROCRAND_PHILOX4x32_DEFAULT_SEED).
typedef philox4x32_10_engine<> philox4x32_10;
/// \typedef xorwow
/// \brief Typedef of \p rocrand_cpp::xorwow_engine PRNG engine with default seed (#ROCRAND_XORWOW_DEFAULT_SEED).
typedef xorwow_engine<> xorwow;
/// \typedef mrg31k3p
/// \brief Typedef of \p rocrand_cpp::mrg31k3p_engine PRNG engine with default seed (#ROCRAND_MRG31K3P_DEFAULT_SEED).
typedef mrg31k3p_engine<> mrg31k3p;
/// \typedef mrg32k3a
/// \brief Typedef of \p rocrand_cpp::mrg32k3a_engine PRNG engine with default seed (#ROCRAND_MRG32K3A_DEFAULT_SEED).
typedef mrg32k3a_engine<> mrg32k3a;
/// \typedef mtgp32
/// \brief Typedef of \p rocrand_cpp::mtgp32_engine PRNG engine with default seed (0).
typedef mtgp32_engine<> mtgp32;
/// \typedef lfsr113
/// \brief Typedef of \p rocrand_cpp::lfsr113_engine PRNG engine with default seed (#ROCRAND_LFSR113_DEFAULT_SEED_X,
/// #ROCRAND_LFSR113_DEFAULT_SEED_Y, #ROCRAND_LFSR113_DEFAULT_SEED_Z, #ROCRAND_LFSR113_DEFAULT_SEED_W).
typedef lfsr113_engine<> lfsr113;
/// \typedef mt19937
/// \brief Typedef of \p rocrand_cpp::mt19937_engine PRNG engine with default seed (0).
typedef mt19937_engine<> mt19937;
/// \typedef threefry2x32
/// \brief Typedef of \p rocrand_cpp::threefry2x32_20_engine PRNG engine with default seed (0).
typedef threefry2x32_20_engine<> threefry2x32;
/// \typedef threefry2x64
/// \brief Typedef of \p rocrand_cpp::threefry2x64_20_engine PRNG engine with default seed (0).
typedef threefry2x64_20_engine<> threefry2x64;
/// \typedef threefry4x32
/// \brief Typedef of \p rocrand_cpp::threefry4x32_20_engine PRNG engine with default seed (0).
typedef threefry4x32_20_engine<> threefry4x32;
/// \typedef threefry4x64
/// \brief Typedef of \p rocrand_cpp::threefry4x64_20_engine PRNG engine with default seed (0).
typedef threefry4x64_20_engine<> threefry4x64;
/// \typedef sobol32
/// \brief Typedef of \p rocrand_cpp::sobol32_engine QRNG engine with default number of dimensions (1).
typedef sobol32_engine<> sobol32;
/// \typedef scrambled_sobol32
/// \brief Typedef of \p rocrand_cpp::scrambled_sobol32_engine QRNG engine with default number of dimensions (1).
typedef scrambled_sobol32_engine<> scrambled_sobol32;
/// \typedef sobol64
/// \brief Typedef of \p rocrand_cpp::sobol64_engine QRNG engine with default number of dimensions (1).
typedef sobol64_engine<> sobol64;
/// \typedef scrambled_sobol64
/// \brief Typedef of \p rocrand_cpp::scrambled_sobol64_engine QRNG engine with default number of dimensions (1).
typedef scrambled_sobol64_engine<> scrambled_sobol64;

/// \typedef default_random_engine
/// \brief Default random engine.
typedef xorwow default_random_engine;

/// \typedef random_device
///
/// \brief A non-deterministic uniform random number generator.
///
/// rocrand_cpp::random_device is non-deterministic uniform random number generator,
/// or a pseudo-random number engine if there is no support for non-deterministic
/// random number generation. It's implemented as a typedef of std::random_device.
///
/// For practical use rocrand_cpp::random_device is generally only used to seed a PRNG
/// such as \ref rocrand_cpp::mtgp32_engine.
///
/// Example:
/// \code
/// #include <rocrand/rocrand.hpp>
///
/// int main()
/// {
///     const size_t size = 8192;
///     float * output;
///     hipMalloc(&output, size * sizeof(float));
///
///     rocrand_cpp::random_device rd;
///     rocrand_cpp::mtgp32 engine(rd()); // seed engine with a real random value, if available
///     rocrand_cpp::normal_distribution<float> dist(0.0, 1.5);
///     dist(engine, output, size);
/// }
/// \endcode
typedef std::random_device random_device;

/// \brief Returns rocRAND version.
/// \return rocRAND version number as an \p int value.
inline int version()
{
    int x;
    rocrand_status status = rocrand_get_version(&x);
    if(status != ROCRAND_STATUS_SUCCESS)
    {
        throw rocrand_cpp::error(status);
    }
    return x;
}

/// @} // end of group rocrandhostcpp

} // end namespace rocrand_cpp

#endif // #if __cplusplus >= 201103L
#endif // ROCRAND_HPP_
