/* ************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights Reserved.
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
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * ************************************************************************ */
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#define _USE_MATH_DEFINES
#endif

#include "host_vector.hpp"
#include "../../utils/allocate_free.hpp"
#include "../../utils/def.hpp"
#include "../../utils/log.hpp"
#include "../../utils/math_functions.hpp"
#include "../base_vector.hpp"
#include "rocalution/version.hpp"

#include <algorithm>
#include <complex>
#include <fstream>
#include <limits>
#include <math.h>
#include <numeric>
#include <typeindex>
#include <typeinfo>

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_set_num_threads(num) ;
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_get_num_threads() 1
#define omp_set_nested(num) ;
#endif

namespace rocalution
{

    // LCOV_EXCL_START
    template <typename ValueType>
    HostVector<ValueType>::HostVector()
    {
        // no default constructors
        LOG_INFO("no default constructor");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    HostVector<ValueType>::HostVector(const Rocalution_Backend_Descriptor& local_backend)
    {
        log_debug(this, "HostVector::HostVector()", "constructor with local_backend");

        this->vec_ = NULL;
        this->set_backend(local_backend);
    }

    template <typename ValueType>
    HostVector<ValueType>::~HostVector()
    {
        log_debug(this, "HostVector::~HostVector()", "destructor");

        this->Clear();
    }

    template <typename ValueType>
    void HostVector<ValueType>::Info(void) const
    {
        LOG_INFO("HostVector<ValueType>, OpenMP threads: " << this->local_backend_.OpenMP_threads);
    }

    template <typename ValueType>
    bool HostVector<ValueType>::Check(void) const
    {
        bool check = true;

        if(this->size_ > 0)
        {
            for(int64_t i = 0; i < this->size_; ++i)
            {
                if((std::abs(this->vec_[i]) == std::numeric_limits<ValueType>::infinity()) || // inf
                   (this->vec_[i] != this->vec_[i]))
                { // NaN
                    // LCOV_EXCL_START
                    LOG_VERBOSE_INFO(2, "*** error: Vector:Check - problems with vector data");
                    return false;
                    // LCOV_EXCL_STOP
                }
            }

            if((std::abs(this->size_) == std::numeric_limits<int64_t>::infinity()) || // inf
               (this->size_ != this->size_))
            { // NaN
                // LCOV_EXCL_START
                LOG_VERBOSE_INFO(2, "*** error: Vector:Check - problems with vector size");
                return false;
                // LCOV_EXCL_STOP
            }
        }
        else
        {
            assert(this->size_ == 0);
            assert(this->vec_ == NULL);
        }

        return check;
    }

    template <>
    bool HostVector<bool>::Check(void) const
    {
        bool check = true;

        if(this->size_ > 0)
        {
            if(std::abs(this->size_) == std::numeric_limits<int64_t>::infinity())
            {
                // inf
                // LCOV_EXCL_START
                LOG_VERBOSE_INFO(2, "*** error: Vector:Check - problems with vector size");
                return false;
                // LCOV_EXCL_STOP
            }
        }
        else
        {
            assert(this->size_ == 0);
            assert(this->vec_ == NULL);
        }

        return check;
    }

    template <typename ValueType>
    void HostVector<ValueType>::Allocate(int64_t n)
    {
        assert(n >= 0);

        this->Clear();

        if(n > 0)
        {
            allocate_host(n, &this->vec_);

            set_to_zero_host(n, this->vec_);

            this->size_ = n;
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::SetDataPtr(ValueType** ptr, int64_t size)
    {
        assert(size >= 0);

        this->Clear();

        if(size > 0)
        {
            assert(*ptr != NULL);
        }

        this->vec_  = *ptr;
        this->size_ = size;
    }

    template <typename ValueType>
    void HostVector<ValueType>::LeaveDataPtr(ValueType** ptr)
    {
        assert(this->size_ >= 0);

        // see free_host function for details
        *ptr       = this->vec_;
        this->vec_ = NULL;

        this->size_ = 0;
    }

    template <typename ValueType>
    void HostVector<ValueType>::CopyFromData(const ValueType* data)
    {
        copy_h2h(this->size_, data, this->vec_);
    }

    template <typename ValueType>
    void HostVector<ValueType>::CopyFromHostData(const ValueType* data)
    {
        // We are already on host, just copy
        this->CopyFromData(data);
    }

    template <typename ValueType>
    void HostVector<ValueType>::CopyToData(ValueType* data) const
    {
        copy_h2h(this->size_, this->vec_, data);
    }

    template <typename ValueType>
    void HostVector<ValueType>::CopyToHostData(ValueType* data) const
    {
        // We are already on host, just copy
        this->CopyToData(data);
    }

    template <typename ValueType>
    void HostVector<ValueType>::Clear(void)
    {
        if(this->size_ > 0)
        {
            free_host(&this->vec_);
            this->size_ = 0;
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::CopyFrom(const BaseVector<ValueType>& vec)
    {
        if(this != &vec)
        {
            if(const HostVector<ValueType>* cast_vec
               = dynamic_cast<const HostVector<ValueType>*>(&vec))
            {
                if(this->size_ == 0)
                {
                    // Allocate local vector
                    this->Allocate(cast_vec->size_);
                }

                assert(cast_vec->size_ == this->size_);

                copy_h2h(this->size_, cast_vec->vec_, this->vec_);
            }
            else
            {
                // non-host type
                vec.CopyTo(this);
            }
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::CopyTo(BaseVector<ValueType>* vec) const
    {
        vec->CopyFrom(*this);
    }

    // LCOV_EXCL_START
    template <typename ValueType>
    void HostVector<ValueType>::CopyFromFloat(const BaseVector<float>& vec)
    {
        LOG_INFO("Mixed precision for non-complex to complex casting is not allowed");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <>
    void HostVector<double>::CopyFromFloat(const BaseVector<float>& vec)
    {
        if(const HostVector<float>* cast_vec = dynamic_cast<const HostVector<float>*>(&vec))
        {
            if(this->size_ == 0)
            {
                // Allocate local vector
                this->Allocate(cast_vec->size_);
            }

            assert(cast_vec->size_ == this->size_);

            _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int64_t i = 0; i < this->size_; ++i)
            {
                this->vec_[i] = static_cast<double>(cast_vec->vec_[i]);
            }
        }
        else
        {
            // LCOV_EXCL_START
            LOG_INFO("No cross backend casting");
            FATAL_ERROR(__FILE__, __LINE__);
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::CopyFromDouble(const BaseVector<double>& vec)
    {
        LOG_INFO("Mixed precision for non-complex to complex casting is not allowed");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <>
    void HostVector<float>::CopyFromDouble(const BaseVector<double>& vec)
    {
        if(const HostVector<double>* cast_vec = dynamic_cast<const HostVector<double>*>(&vec))
        {
            if(this->size_ == 0)
            {
                // Allocate local vector
                this->Allocate(cast_vec->size_);
            }

            assert(cast_vec->size_ == this->size_);

            _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
            for(int64_t i = 0; i < this->size_; ++i)
            {
                this->vec_[i] = static_cast<float>(cast_vec->vec_[i]);
            }
        }
        else
        {
            // LCOV_EXCL_START
            LOG_INFO("No cross backend casting");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::Zeros(void)
    {
        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i] = static_cast<ValueType>(0);
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::Ones(void)
    {
        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i] = static_cast<ValueType>(1);
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::SetValues(ValueType val)
    {
        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i] = val;
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::SetRandomUniform(unsigned long long seed, ValueType a, ValueType b)
    {
        assert(a <= b);

        // Fill this with random data from interval [a,b]
        srand(seed);
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i]
                = a + static_cast<ValueType>(rand()) / static_cast<ValueType>(RAND_MAX) * (b - a);
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::SetRandomNormal(unsigned long long seed,
                                                ValueType          mean,
                                                ValueType          var)
    {
        srand(seed);
        for(int64_t i = 0; i < this->size_; ++i)
        {
            // Box-Muller
            ValueType u1 = static_cast<ValueType>(rand()) / static_cast<ValueType>(RAND_MAX);
            ValueType u2 = static_cast<ValueType>(rand()) / static_cast<ValueType>(RAND_MAX);

            this->vec_[i] = sqrt(static_cast<ValueType>(-2) * log(u1))
                            * cos(static_cast<ValueType>(2 * M_PI) * u2);

            // Shift
            this->vec_[i] = mean + var * this->vec_[i];
        }
    }

    // LCOV_EXCL_START
    template <>
    void HostVector<bool>::SetRandomNormal(unsigned long long seed, bool mean, bool var)
    {
        LOG_INFO("What is bool HostVector<ValueType>::SetRandomNormal(void) const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    void HostVector<ValueType>::ReadFileASCII(const std::string& filename)
    {
        std::ifstream file;
        std::string   line;
        int64_t       n = 0;

        LOG_INFO("ReadFileASCII: filename=" << filename << "; reading...");

        file.open((char*)filename.c_str(), std::ifstream::in);

        if(!file.is_open())
        {
            // LCOV_EXCL_START
            LOG_INFO("Can not open vector file [read]:" << filename);
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        this->Clear();

        // get the size of the vector
        while(std::getline(file, line))
        {
            ++n;
        }

        this->Allocate(n);

        file.clear();
        file.seekg(0, std::ios_base::beg);

        for(int64_t i = 0; i < n; ++i)
        {
            file >> this->vec_[i];
        }

        file.close();

        LOG_INFO("ReadFileASCII: filename=" << filename << "; done");
    }

    template <typename ValueType>
    void HostVector<ValueType>::WriteFileASCII(const std::string& filename) const
    {
        std::ofstream file;
        std::string   line;

        LOG_INFO("WriteFileASCII: filename=" << filename << "; writing...");

        file.open((char*)filename.c_str(), std::ifstream::out);

        if(!file.is_open())
        {
            // LCOV_EXCL_START
            LOG_INFO("Can not open vector file [write]:" << filename);
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        file.setf(std::ios::scientific);

        for(int64_t n = 0; n < this->size_; n++)
        {
            file << this->vec_[n] << std::endl;
        }

        file.close();

        LOG_INFO("WriteFileASCII: filename=" << filename << "; done");
    }

    template <typename ValueType>
    void HostVector<ValueType>::ReadFileBinary(const std::string& filename)
    {
        LOG_INFO("ReadFileBinary: filename=" << filename << "; reading...");

        std::ifstream in(filename.c_str(), std::ios::in | std::ios::binary);

        if(!in.is_open())
        {
            // LCOV_EXCL_START
            LOG_INFO("ReadFileBinary: filename=" << filename << "; cannot open file");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        // Header
        std::string header;
        std::getline(in, header);

        if(header != "#rocALUTION binary vector file")
        {
            // LCOV_EXCL_START
            LOG_INFO("ReadFileBinary: filename=" << filename << " is not a rocALUTION vector");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        // rocALUTION version
        int version;
        in.read((char*)&version, sizeof(int));

        // Read size
        int64_t n;

        // We need backward compatibility, v3.0.0 and later will store sizes with 64 bits
        if(version < 30000)
        {
            int size32;

            in.read((char*)&size32, sizeof(int));

            n = size32;
        }
        else
        {
            in.read((char*)&n, sizeof(int64_t));
        }

        //        if(version != __ROCALUTION_VER)
        //        {
        //            LOG_INFO("ReadFileBinary: file version mismatch");
        //            return false;
        //        }

        this->Clear();
        this->Allocate(n);

        // We read always in double precision
        if(typeid(ValueType) == typeid(double))
        {
            in.read((char*)this->vec_, sizeof(ValueType) * n);
        }
        else if(typeid(ValueType) == typeid(float))
        {
            std::vector<double> tmp(n);

            in.read((char*)tmp.data(), sizeof(double) * n);

            for(int64_t i = 0; i < n; ++i)
            {
                this->vec_[i] = static_cast<ValueType>(tmp[i]);
            }
        }
        else if(typeid(ValueType) == typeid(int) || typeid(ValueType) == typeid(int64_t))
        {
            in.read((char*)this->vec_, sizeof(ValueType) * n);
        }
        else
        {
            // LCOV_EXCL_START
            LOG_INFO("ReadFileBinary: filename=" << filename << "; internal error");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        // Check ifstream status
        if(!in)
        {
            // LCOV_EXCL_START
            LOG_INFO("ReadFileBinary: filename=" << filename << "; could not read from file");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        in.close();

        LOG_INFO("ReadFileBinary: filename=" << filename << "; done");
    }

    template <typename ValueType>
    void HostVector<ValueType>::WriteFileBinary(const std::string& filename) const
    {
        LOG_INFO("WriteFileBinary: filename=" << filename << "; writing...");

        std::ofstream out(filename.c_str(), std::ios::out | std::ios::binary);

        if(!out.is_open())
        {
            // LCOV_EXCL_START
            LOG_INFO("WriteFileBinary: filename=" << filename << "; cannot open file");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        // Header
        out << "#rocALUTION binary vector file" << std::endl;

        // rocALUTION version
        int version = __ROCALUTION_VER;
        out.write((char*)&version, sizeof(int));

        // Data
        out.write((char*)&this->size_, sizeof(int64_t));

        // We write always in double precision
        if(typeid(ValueType) == typeid(double))
        {
            out.write((char*)this->vec_, sizeof(ValueType) * this->size_);
        }
        else if(typeid(ValueType) == typeid(float))
        {
            std::vector<double> tmp(this->size_);

            for(int64_t i = 0; i < this->size_; ++i)
            {
                tmp[i] = rocalution_double(this->vec_[i]);
            }

            out.write((char*)tmp.data(), sizeof(double) * this->size_);
        }
        else if(typeid(ValueType) == typeid(int) || typeid(ValueType) == typeid(int64_t))
        {
            out.write((char*)this->vec_, sizeof(ValueType) * this->size_);
        }
        else // TODO complex
        {
            // LCOV_EXCL_START
            LOG_INFO("WriteFileBinary: filename=" << filename << "; internal error");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        // Check ofstream status
        if(!out)
        {
            // LCOV_EXCL_START
            LOG_INFO("ReadFileBinary: filename=" << filename << "; could not write to file");
            FATAL_ERROR(__FILE__, __LINE__);
            // LCOV_EXCL_STOP
        }

        out.close();

        LOG_INFO("WriteFileBinary: filename=" << filename << "; done");
    }

    template <typename ValueType>
    void HostVector<ValueType>::AddScale(const BaseVector<ValueType>& x, ValueType alpha)
    {
        const HostVector<ValueType>* cast_x = dynamic_cast<const HostVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i] = this->vec_[i] + alpha * cast_x->vec_[i];
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::ScaleAdd(ValueType alpha, const BaseVector<ValueType>& x)
    {
        const HostVector<ValueType>* cast_x = dynamic_cast<const HostVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i] = alpha * this->vec_[i] + cast_x->vec_[i];
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::ScaleAddScale(ValueType                    alpha,
                                              const BaseVector<ValueType>& x,
                                              ValueType                    beta)
    {
        const HostVector<ValueType>* cast_x = dynamic_cast<const HostVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i] = alpha * this->vec_[i] + beta * cast_x->vec_[i];
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::ScaleAddScale(ValueType                    alpha,
                                              const BaseVector<ValueType>& x,
                                              ValueType                    beta,
                                              int64_t                      src_offset,
                                              int64_t                      dst_offset,
                                              int64_t                      size)
    {
        const HostVector<ValueType>* cast_x = dynamic_cast<const HostVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ > 0);
        assert(cast_x->size_ > 0);
        assert(size > 0);
        assert(src_offset + size <= cast_x->size_);
        assert(dst_offset + size <= this->size_);

        _set_omp_backend_threads(this->local_backend_, size);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < size; ++i)
        {
            this->vec_[i + dst_offset]
                = alpha * this->vec_[i + dst_offset] + beta * cast_x->vec_[i + src_offset];
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::ScaleAdd2(ValueType                    alpha,
                                          const BaseVector<ValueType>& x,
                                          ValueType                    beta,
                                          const BaseVector<ValueType>& y,
                                          ValueType                    gamma)
    {
        const HostVector<ValueType>* cast_x = dynamic_cast<const HostVector<ValueType>*>(&x);
        const HostVector<ValueType>* cast_y = dynamic_cast<const HostVector<ValueType>*>(&y);

        assert(cast_x != NULL);
        assert(cast_y != NULL);
        assert(this->size_ == cast_x->size_);
        assert(this->size_ == cast_y->size_);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i]
                = alpha * this->vec_[i] + beta * cast_x->vec_[i] + gamma * cast_y->vec_[i];
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::Scale(ValueType alpha)
    {
        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i] *= alpha;
        }
    }

    // LCOV_EXCL_START
    template <>
    void HostVector<bool>::Scale(bool alpha)
    {
        LOG_INFO("What is bool HostVector<ValueType>::Scale(void) const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    ValueType HostVector<ValueType>::Dot(const BaseVector<ValueType>& x) const
    {
        const HostVector<ValueType>* cast_x = dynamic_cast<const HostVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        ValueType dot = static_cast<ValueType>(0);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : dot)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            dot += this->vec_[i] * cast_x->vec_[i];
        }

        return dot;
    }

    template <>
    std::complex<float>
        HostVector<std::complex<float>>::Dot(const BaseVector<std::complex<float>>& x) const
    {
        const HostVector<std::complex<float>>* cast_x
            = dynamic_cast<const HostVector<std::complex<float>>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        float dot_real = 0.0f;
        float dot_imag = 0.0f;

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : dot_real, dot_imag)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            dot_real += this->vec_[i].real() * cast_x->vec_[i].real()
                        + this->vec_[i].imag() * cast_x->vec_[i].imag();
            dot_imag += this->vec_[i].real() * cast_x->vec_[i].imag()
                        - this->vec_[i].imag() * cast_x->vec_[i].real();
        }

        return std::complex<float>(dot_real, dot_imag);
    }

    template <>
    std::complex<double>
        HostVector<std::complex<double>>::Dot(const BaseVector<std::complex<double>>& x) const
    {
        const HostVector<std::complex<double>>* cast_x
            = dynamic_cast<const HostVector<std::complex<double>>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        double dot_real = 0.0;
        double dot_imag = 0.0;

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : dot_real, dot_imag)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            dot_real += this->vec_[i].real() * cast_x->vec_[i].real()
                        + this->vec_[i].imag() * cast_x->vec_[i].imag();
            dot_imag += this->vec_[i].real() * cast_x->vec_[i].imag()
                        - this->vec_[i].imag() * cast_x->vec_[i].real();
        }

        return std::complex<double>(dot_real, dot_imag);
    }

    template <typename ValueType>
    ValueType HostVector<ValueType>::DotNonConj(const BaseVector<ValueType>& x) const
    {
        return this->Dot(x);
    }

    template <>
    std::complex<float>
        HostVector<std::complex<float>>::DotNonConj(const BaseVector<std::complex<float>>& x) const
    {
        const HostVector<std::complex<float>>* cast_x
            = dynamic_cast<const HostVector<std::complex<float>>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        float dot_real = 0.0f;
        float dot_imag = 0.0f;

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : dot_real, dot_imag)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            dot_real += this->vec_[i].real() * cast_x->vec_[i].real()
                        - this->vec_[i].imag() * cast_x->vec_[i].imag();
            dot_imag += this->vec_[i].real() * cast_x->vec_[i].imag()
                        + this->vec_[i].imag() * cast_x->vec_[i].real();
        }

        return std::complex<float>(dot_real, dot_imag);
    }

    template <>
    std::complex<double> HostVector<std::complex<double>>::DotNonConj(
        const BaseVector<std::complex<double>>& x) const
    {
        const HostVector<std::complex<double>>* cast_x
            = dynamic_cast<const HostVector<std::complex<double>>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        double dot_real = 0.0;
        double dot_imag = 0.0;

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : dot_real, dot_imag)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            dot_real += this->vec_[i].real() * cast_x->vec_[i].real()
                        - this->vec_[i].imag() * cast_x->vec_[i].imag();
            dot_imag += this->vec_[i].real() * cast_x->vec_[i].imag()
                        + this->vec_[i].imag() * cast_x->vec_[i].real();
        }

        return std::complex<double>(dot_real, dot_imag);
    }

    template <typename ValueType>
    ValueType HostVector<ValueType>::Asum(void) const
    {
        ValueType asum = static_cast<ValueType>(0);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : asum)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            asum += std::abs(this->vec_[i]);
        }

        return asum;
    }

    template <>
    std::complex<float> HostVector<std::complex<float>>::Asum(void) const
    {
        float asum_real = 0.0f;
        float asum_imag = 0.0f;

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : asum_real, asum_imag)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            asum_real += std::abs(this->vec_[i].real());
            asum_imag += std::abs(this->vec_[i].imag());
        }

        return std::complex<float>(asum_real, asum_imag);
    }

    template <>
    std::complex<double> HostVector<std::complex<double>>::Asum(void) const
    {
        double asum_real = 0.0;
        double asum_imag = 0.0;

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : asum_real, asum_imag)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            asum_real += std::abs(this->vec_[i].real());
            asum_imag += std::abs(this->vec_[i].imag());
        }

        return std::complex<double>(asum_real, asum_imag);
    }

    // LCOV_EXCL_START
    template <>
    bool HostVector<bool>::Asum(void) const
    {
        LOG_INFO("What is bool HostVector<ValueType>::Asum(void) const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    int64_t HostVector<ValueType>::Amax(ValueType& value) const
    {
        int64_t index = 0;

        value = static_cast<ValueType>(0);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            ValueType val = std::abs(this->vec_[i]);
            if(val > value)
#ifdef _OPENMP
#pragma omp critical
#endif
            {
                if(val > value)
                {
                    value = val;
                    index = i;
                }
            }
        }

        return index;
    }

    // LCOV_EXCL_START
    template <>
    int64_t HostVector<bool>::Amax(bool& value) const
    {
        LOG_INFO("What is int64_t HostVector<ValueType>::Amax(void) const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    ValueType HostVector<ValueType>::Norm(void) const
    {
        ValueType norm2 = static_cast<ValueType>(0);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : norm2)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            norm2 += this->vec_[i] * this->vec_[i];
        }

        return std::sqrt(norm2);
    }

    template <>
    std::complex<float> HostVector<std::complex<float>>::Norm(void) const
    {
        float norm2 = 0.0f;

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : norm2)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            norm2 += this->vec_[i].real() * this->vec_[i].real()
                     + this->vec_[i].imag() * this->vec_[i].imag();
        }

        std::complex<float> res(sqrt(norm2), 0.0f);

        return res;
    }

    template <>
    std::complex<double> HostVector<std::complex<double>>::Norm(void) const
    {
        double norm2 = 0.0;

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : norm2)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            norm2 += this->vec_[i].real() * this->vec_[i].real()
                     + this->vec_[i].imag() * this->vec_[i].imag();
        }

        std::complex<double> res(sqrt(norm2), 0.0);

        return res;
    }

    // LCOV_EXCL_START
    template <>
    int HostVector<int>::Norm(void) const
    {
        LOG_INFO("What is int HostVector<ValueType>::Norm(void) const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    int64_t HostVector<int64_t>::Norm(void) const
    {
        LOG_INFO("What is int64_t HostVector<ValueType>::Norm(void) const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    bool HostVector<bool>::Norm(void) const
    {
        LOG_INFO("What is bool HostVector<ValueType>::Norm(void) const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    ValueType HostVector<ValueType>::Reduce(void) const
    {
        ValueType reduce = static_cast<ValueType>(0);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : reduce)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            reduce += this->vec_[i];
        }

        return reduce;
    }

    template <>
    std::complex<float> HostVector<std::complex<float>>::Reduce(void) const
    {
        float reduce_real = 0.0f;
        float reduce_imag = 0.0f;

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : reduce_real, reduce_imag)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            reduce_real += this->vec_[i].real();
            reduce_imag += this->vec_[i].imag();
        }

        return std::complex<float>(reduce_real, reduce_imag);
    }

    template <>
    std::complex<double> HostVector<std::complex<double>>::Reduce(void) const
    {
        double reduce_real = 0.0;
        double reduce_imag = 0.0;

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for reduction(+ : reduce_real, reduce_imag)
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            reduce_real += this->vec_[i].real();
            reduce_imag += this->vec_[i].imag();
        }

        return std::complex<double>(reduce_real, reduce_imag);
    }

    template <typename ValueType>
    ValueType HostVector<ValueType>::InclusiveSum(const BaseVector<ValueType>& vec)
    {
        const HostVector<ValueType>* cast_vec = dynamic_cast<const HostVector<ValueType>*>(&vec);

        assert(cast_vec != NULL);
        assert(this->size_ == cast_vec->size_);

        if(this->size_ == 0)
        {
            return 0;
        }

        if(this->vec_ == cast_vec->vec_)
        {
            for(int64_t i = 1; i < this->size_; ++i)
            {
                this->vec_[i] += this->vec_[i - 1];
            }
        }
        else
        {
            this->vec_[0] = cast_vec->vec_[0];

            for(int64_t i = 1; i < this->size_; ++i)
            {
                this->vec_[i] = this->vec_[i - 1] + cast_vec->vec_[i];
            }
        }

        return this->vec_[this->size_ - 1];
    }

    template <typename ValueType>
    ValueType HostVector<ValueType>::ExclusiveSum(const BaseVector<ValueType>& vec)
    {
        const HostVector<ValueType>* cast_vec = dynamic_cast<const HostVector<ValueType>*>(&vec);

        assert(cast_vec != NULL);
        assert(this->size_ == cast_vec->size_);

        if(this->size_ == 0)
        {
            return 0;
        }

        if(this->vec_ == cast_vec->vec_)
        {
            for(int64_t i = this->size_ - 1; i > 0; --i)
            {
                this->vec_[i] = this->vec_[i - 1];
            }

            this->vec_[0] = 0;

            for(int64_t i = 1; i < this->size_; ++i)
            {
                this->vec_[i] += this->vec_[i - 1];
            }
        }
        else
        {
            this->vec_[0] = 0;

            for(int64_t i = 0; i < this->size_ - 1; ++i)
            {
                this->vec_[i + 1] = this->vec_[i] + cast_vec->vec_[i];
            }
        }

        return this->vec_[this->size_ - 1];
    }

    template <typename ValueType>
    void HostVector<ValueType>::PointWiseMult(const BaseVector<ValueType>& x)
    {
        const HostVector<ValueType>* cast_x = dynamic_cast<const HostVector<ValueType>*>(&x);

        assert(cast_x != NULL);
        assert(this->size_ == cast_x->size_);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i] = this->vec_[i] * cast_x->vec_[i];
        }
    }

    // LCOV_EXCL_START
    template <>
    void HostVector<bool>::PointWiseMult(const BaseVector<bool>& x)
    {
        LOG_INFO("What is bool HostVector<ValueType>::PointWiseMult() const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    void HostVector<ValueType>::PointWiseMult(const BaseVector<ValueType>& x,
                                              const BaseVector<ValueType>& y)
    {
        const HostVector<ValueType>* cast_x = dynamic_cast<const HostVector<ValueType>*>(&x);
        const HostVector<ValueType>* cast_y = dynamic_cast<const HostVector<ValueType>*>(&y);

        assert(cast_x != NULL);
        assert(cast_y != NULL);
        assert(this->size_ == cast_x->size_);
        assert(this->size_ == cast_y->size_);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i] = cast_y->vec_[i] * cast_x->vec_[i];
        }
    }

    // LCOV_EXCL_START
    template <>
    void HostVector<bool>::PointWiseMult(const BaseVector<bool>& x, const BaseVector<bool>& y)
    {
        LOG_INFO("What is bool HostVector<ValueType>::PointWiseMult() const?");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <typename ValueType>
    void HostVector<ValueType>::CopyFrom(const BaseVector<ValueType>& src,
                                         int64_t                      src_offset,
                                         int64_t                      dst_offset,
                                         int64_t                      size)
    {
        const HostVector<ValueType>* cast_src = dynamic_cast<const HostVector<ValueType>*>(&src);

        assert(cast_src != NULL);
        // TOOD check always for == this?
        assert(&src != this);
        assert(this->size_ > 0);
        assert(cast_src->size_ > 0);
        assert(size > 0);
        assert(src_offset + size <= cast_src->size_);
        assert(dst_offset + size <= this->size_);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < size; ++i)
        {
            this->vec_[i + dst_offset] = cast_src->vec_[i + src_offset];
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::Permute(const BaseVector<int>& permutation)
    {
        const HostVector<int>* cast_perm = dynamic_cast<const HostVector<int>*>(&permutation);

        assert(cast_perm != NULL);
        assert(this->size_ == cast_perm->size_);

        HostVector<ValueType> vec_tmp(this->local_backend_);
        vec_tmp.Allocate(this->size_);
        vec_tmp.CopyFrom(*this);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            assert_dbg(cast_perm->vec_[i] >= 0);
            assert_dbg(cast_perm->vec_[i] < this->size_);
            this->vec_[cast_perm->vec_[i]] = vec_tmp.vec_[i];
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::PermuteBackward(const BaseVector<int>& permutation)
    {
        const HostVector<int>* cast_perm = dynamic_cast<const HostVector<int>*>(&permutation);

        assert(cast_perm != NULL);
        assert(this->size_ == cast_perm->size_);

        HostVector<ValueType> vec_tmp(this->local_backend_);
        vec_tmp.Allocate(this->size_);
        vec_tmp.CopyFrom(*this);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            assert_dbg(cast_perm->vec_[i] >= 0);
            assert_dbg(cast_perm->vec_[i] < this->size_);
            this->vec_[i] = vec_tmp.vec_[cast_perm->vec_[i]];
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::CopyFromPermute(const BaseVector<ValueType>& src,
                                                const BaseVector<int>&       permutation)
    {
        assert(this != &src);

        const HostVector<ValueType>* cast_vec  = dynamic_cast<const HostVector<ValueType>*>(&src);
        const HostVector<int>*       cast_perm = dynamic_cast<const HostVector<int>*>(&permutation);
        assert(cast_perm != NULL);
        assert(cast_vec != NULL);

        assert(cast_vec->size_ == this->size_);
        assert(cast_perm->size_ == this->size_);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[cast_perm->vec_[i]] = cast_vec->vec_[i];
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::CopyFromPermuteBackward(const BaseVector<ValueType>& src,
                                                        const BaseVector<int>&       permutation)
    {
        assert(this != &src);

        const HostVector<ValueType>* cast_vec  = dynamic_cast<const HostVector<ValueType>*>(&src);
        const HostVector<int>*       cast_perm = dynamic_cast<const HostVector<int>*>(&permutation);
        assert(cast_perm != NULL);
        assert(cast_vec != NULL);

        assert(cast_vec->size_ == this->size_);
        assert(cast_perm->size_ == this->size_);

        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i] = cast_vec->vec_[cast_perm->vec_[i]];
        }
    }

    template <typename ValueType>
    bool HostVector<ValueType>::Restriction(const BaseVector<ValueType>& vec_fine,
                                            const BaseVector<int>&       map)
    {
        assert(this != &vec_fine);

        const HostVector<ValueType>* cast_vec
            = dynamic_cast<const HostVector<ValueType>*>(&vec_fine);
        const HostVector<int>* cast_map = dynamic_cast<const HostVector<int>*>(&map);
        assert(cast_map != NULL);
        assert(cast_vec != NULL);
        assert(cast_map->size_ == cast_vec->size_);

        this->Zeros();

        for(int64_t i = 0; i < cast_vec->size_; ++i)
        {
            if(cast_map->vec_[i] != -1)
            {
                this->vec_[cast_map->vec_[i]] += cast_vec->vec_[i];
            }
        }

        return true;
    }

    template <typename ValueType>
    bool HostVector<ValueType>::Prolongation(const BaseVector<ValueType>& vec_coarse,
                                             const BaseVector<int>&       map)
    {
        assert(this != &vec_coarse);

        const HostVector<ValueType>* cast_vec
            = dynamic_cast<const HostVector<ValueType>*>(&vec_coarse);
        const HostVector<int>* cast_map = dynamic_cast<const HostVector<int>*>(&map);
        assert(cast_map != NULL);
        assert(cast_vec != NULL);
        assert(cast_map->size_ == this->size_);

        for(int64_t i = 0; i < this->size_; ++i)
        {
            if(cast_map->vec_[i] != -1)
            {
                this->vec_[i] = cast_vec->vec_[cast_map->vec_[i]];
            }
            else
            {
                this->vec_[i] = static_cast<ValueType>(0);
            }
        }

        return true;
    }

    template <typename ValueType>
    void HostVector<ValueType>::GetIndexValues(const BaseVector<int>& index,
                                               BaseVector<ValueType>* values) const
    {
        assert(values != NULL);

        const HostVector<int>* cast_idx = dynamic_cast<const HostVector<int>*>(&index);
        HostVector<ValueType>* cast_vec = dynamic_cast<HostVector<ValueType>*>(values);

        assert(cast_idx != NULL);
        assert(cast_vec != NULL);
        assert(cast_vec->size_ == cast_idx->size_);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < cast_idx->size_; ++i)
        {
            cast_vec->vec_[i] = this->vec_[cast_idx->vec_[i]];
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::SetIndexValues(const BaseVector<int>&       index,
                                               const BaseVector<ValueType>& values)
    {
        const HostVector<int>*       cast_idx = dynamic_cast<const HostVector<int>*>(&index);
        const HostVector<ValueType>* cast_vec = dynamic_cast<const HostVector<ValueType>*>(&values);

        assert(cast_idx != NULL);
        assert(cast_vec != NULL);
        assert(cast_vec->size_ == cast_idx->size_);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1024)
#endif
        for(int i = 0; i < cast_idx->size_; ++i)
        {
            this->vec_[cast_idx->vec_[i]] = cast_vec->vec_[i];
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::AddIndexValues(const BaseVector<int>&       index,
                                               const BaseVector<ValueType>& values)
    {
        const HostVector<int>*       cast_idx = dynamic_cast<const HostVector<int>*>(&index);
        const HostVector<ValueType>* cast_vec = dynamic_cast<const HostVector<ValueType>*>(&values);

        assert(cast_idx != NULL);
        assert(cast_vec != NULL);
        assert(cast_vec->size_ == cast_idx->size_);

#ifdef _OPENMP
#pragma parallel for schuedule(dynamic, 1024)
#endif
        for(int i = 0; i < cast_idx->size_; ++i)
        {
            this->vec_[cast_idx->vec_[i]] += cast_vec->vec_[i];
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::GetContinuousValues(int64_t    start,
                                                    int64_t    end,
                                                    ValueType* values) const
    {
        assert(start >= 0);
        assert(end >= start);
        assert(end <= this->size_);

        copy_h2h(end - start, this->vec_ + start, values);
    }

    template <typename ValueType>
    void HostVector<ValueType>::SetContinuousValues(int64_t          start,
                                                    int64_t          end,
                                                    const ValueType* values)
    {
        assert(start >= 0);
        assert(end >= start);
        assert(end <= this->size_);

        copy_h2h(end - start, values, this->vec_ + start);
    }

    template <typename ValueType>
    void HostVector<ValueType>::RSPMISUpdateCFmap(const BaseVector<int>& index,
                                                  BaseVector<ValueType>* values)
    {
        assert(values != NULL);

        const HostVector<int>* cast_idx = dynamic_cast<const HostVector<int>*>(&index);
        HostVector<ValueType>* cast_vec = dynamic_cast<HostVector<ValueType>*>(values);

        assert(cast_idx != NULL);
        assert(cast_vec != NULL);
        assert(cast_vec->size_ == cast_idx->size_);

#ifdef _OPENMP
#pragma parallel for schuedule(dynamic, 1024)
#endif
        for(int i = 0; i < cast_idx->size_; ++i)
        {
            if(cast_vec->vec_[i] == static_cast<ValueType>(0))
            {
                // Update
                this->vec_[cast_idx->vec_[i]] = static_cast<ValueType>(0);
            }
            else
            {
                // Pack
                cast_vec->vec_[i] = this->vec_[cast_idx->vec_[i]];
            }
        }
    }

    // LCOV_EXCL_START
    template <typename ValueType>
    void HostVector<ValueType>::ExtractCoarseMapping(
        int64_t start, int64_t end, const int* index, int nc, int* size, int* map) const
    {
        LOG_INFO("double/float HostVector<ValueType>::ExtractCoarseMapping() not available");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <>
    void HostVector<int>::ExtractCoarseMapping(
        int64_t start, int64_t end, const int* index, int nc, int* size, int* map) const
    {
        assert(index != NULL);
        assert(size != NULL);
        assert(map != NULL);
        assert(start >= 0);
        assert(end >= start);

        int  ind   = 0;
        int  k     = 0;
        int* check = NULL;
        allocate_host(nc, &check);

        for(int i = 0; i < nc; ++i)
        {
            check[i] = -1;
        }

        // Loop over fine boundary points
        for(int64_t i = start; i < end; ++i)
        {
            int coarse_index = this->vec_[index[i]];

            if(check[coarse_index] == -1)
            {
                map[ind++]          = k;
                check[coarse_index] = k++;
            }
            else
            {
                map[ind++] = check[coarse_index];
            }
        }

        free_host(&check);

        *size = ind;
    }

    // LCOV_EXCL_START
    template <typename ValueType>
    void HostVector<ValueType>::ExtractCoarseBoundary(
        int64_t start, int64_t end, const int* index, int nc, int* size, int* boundary) const
    {
        LOG_INFO("double/float HostVector<ValueType>::ExtractCoarseBoundary() not available");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template <>
    void HostVector<int>::ExtractCoarseBoundary(
        int64_t start, int64_t end, const int* index, int nc, int* size, int* boundary) const
    {
        assert(index != NULL);
        assert(size != NULL);
        assert(boundary != NULL);
        assert(start >= 0);
        assert(end >= start);

        int  ind   = *size;
        int* check = NULL;
        allocate_host(nc, &check);
        set_to_zero_host(nc, check);

        // Loop over fine boundary points
        for(int64_t i = start; i < end; ++i)
        {
            int coarse_index = this->vec_[index[i]];

            if(coarse_index == -1)
            {
                continue;
            }

            if(check[coarse_index] == 0)
            {
                boundary[ind++]     = coarse_index;
                check[coarse_index] = 1;
            }
        }

        free_host(&check);

        *size = ind;
    }

    template <typename ValueType>
    void HostVector<ValueType>::Power(double power)
    {
        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i] = pow(this->vec_[i], static_cast<ValueType>(power));
        }
    }

    template <>
    void HostVector<std::complex<float>>::Power(double power)
    {
        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            this->vec_[i] = pow(this->vec_[i], std::complex<float>(static_cast<float>(power)));
        }
    }

    template <>
    void HostVector<int>::Power(double power)
    {
        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            int value = 1;
            for(int j = 0; j < power; ++j)
            {
                value *= this->vec_[i];
            }

            this->vec_[i] = value;
        }
    }

    template <>
    void HostVector<int64_t>::Power(double power)
    {
        _set_omp_backend_threads(this->local_backend_, this->size_);

#ifdef _OPENMP
#pragma omp parallel for
#endif
        for(int64_t i = 0; i < this->size_; ++i)
        {
            int64_t value = 1;
            for(int j = 0; j < power; ++j)
            {
                value *= this->vec_[i];
            }

            this->vec_[i] = value;
        }
    }

    template <typename ValueType>
    void HostVector<ValueType>::Sort(BaseVector<ValueType>* sorted, BaseVector<int>* perm) const
    {
        if(this->size_ > 0)
        {
            assert(sorted != NULL);

            HostVector<ValueType>* cast_sort = dynamic_cast<HostVector<ValueType>*>(sorted);
            HostVector<int>*       cast_perm = dynamic_cast<HostVector<int>*>(perm);

            assert(cast_sort != NULL);
            assert(cast_sort->size_ >= this->size_);

            if(cast_perm == NULL)
            {
                // Sort without permutation
                copy_h2h(this->size_, this->vec_, cast_sort->vec_);
                std::sort(cast_sort->vec_, cast_sort->vec_ + this->size_);
            }
            else
            {
                assert(cast_perm != NULL);
                assert(cast_perm->size_ >= this->size_);

                // Create identity permutation
                std::iota(cast_perm->vec_, cast_perm->vec_ + this->size_, 0);

                // Sort with permutation
                std::sort(
                    cast_perm->vec_,
                    cast_perm->vec_ + this->size_,
                    [&](const int& a, const int& b) { return (this->vec_[a] < this->vec_[b]); });

                for(int i = 0; i < this->size_; ++i)
                {
                    cast_sort->vec_[i] = this->vec_[cast_perm->vec_[i]];
                }
            }
        }
    }

    // LCOV_EXCL_START
    template <>
    void HostVector<std::complex<float>>::Sort(BaseVector<std::complex<float>>* sorted,
                                               BaseVector<int>*                 perm) const
    {
        LOG_INFO("HostVector::Sort(), how to sort complex numbers?");
        FATAL_ERROR(__FILE__, __LINE__);
    }

    template <>
    void HostVector<std::complex<double>>::Sort(BaseVector<std::complex<double>>* sorted,
                                                BaseVector<int>*                  perm) const
    {
        LOG_INFO("HostVector::Sort(), how to sort complex numbers?");
        FATAL_ERROR(__FILE__, __LINE__);
    }
    // LCOV_EXCL_STOP

    template class HostVector<bool>;
    template class HostVector<double>;
    template class HostVector<float>;
#ifdef SUPPORT_COMPLEX
    template class HostVector<std::complex<double>>;
    template class HostVector<std::complex<float>>;
#endif

    template class HostVector<int>;
    template class HostVector<int64_t>;

} // namespace rocalution
