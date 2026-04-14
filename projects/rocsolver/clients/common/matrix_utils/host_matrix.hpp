/* **************************************************************************
 * Copyright (C) 2018-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * *************************************************************************/

#pragma once

#include <complex>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <iostream>
#include <memory>
#include <new>
#include <type_traits>
#include <vector>

#include "matrix_interface.hpp"
#include "matrix_utils_detail.hpp"

namespace matxu
{
template <typename T_, typename I_ = std::int32_t>
class HostMatrix : public MatrixInterface<T_, I_>
{
public:
    using T = typename MatrixInterface<T_, I_>::T;
    using I = typename MatrixInterface<T_, I_>::I;
    using S = typename MatrixInterface<T_, I_>::S;

    static auto Wrap(T* in_data, I nrows, I ncols) noexcept -> std::unique_ptr<HostMatrix<T_, I_>>
    {
        if((nrows < 1) || (ncols < 1))
        {
            return nullptr;
        }

        auto ptr = std::unique_ptr<HostMatrix<T_, I_>>(
            new(std::nothrow) HostMatrix<T_, I_>(in_data, nrows, ncols));

        if(ptr)
        {
            ptr->is_a_wrapper_ = true;
        }

        return ptr;
    }

    /* static auto Clone(T* in_data, I nrows, I ncols) noexcept -> std::unique_ptr<HostMatrix<T, I>> */
    /* { */
    /*     if ((nrows < 1) || (ncols < 1)) */
    /*     { */
    /*         return nullptr; */
    /*     } */

    /*     auto ptr = std::unique_ptr<HostMatrix<T, I>>(new(std::nothrow) HostMatrix<T, I>(nullptr, nrows, ncols)); */

    /*     ptr->data_ = ptr->cgc_.alloc_and_copy(ptr->size(), in_data); */
    /*     if (ptr->data_ == nullptr) */
    /*     { */
    /*         // If ptr->data_ was not initialized, then ptr does not point to a valid matrix */
    /*         ptr = nullptr; */
    /*     } */

    /*     return ptr; */
    /* } */

    template <typename S_>
    static auto Convert(S_* in_data, I nrows, I ncols) noexcept -> std::unique_ptr<HostMatrix<T, I>>
    {
        if((nrows < 1) || (ncols < 1))
        {
            return nullptr;
        }

        auto ptr = std::unique_ptr<HostMatrix<T, I>>(new(std::nothrow)
                                                         HostMatrix<T, I>(nullptr, nrows, ncols));

        ptr->data_ = ptr->cgc_.alloc(ptr->size());
        if(ptr->data_ == nullptr)
        {
            // If ptr->data_ was not initialized, then ptr does not point to a valid matrix
            ptr = nullptr;
        }

        for(I i = 0; i < ptr->size(); ++i)
        {
            ptr->operator[](i) = T(in_data[i]);
        }

        return ptr;
    }

    template <template <typename, typename> class HostMatrix_, typename TT_, typename II_>
    static auto Convert(const HostMatrix_<TT_, II_>& In) -> HostMatrix<T_, I_>
    {
        HostMatrix<T_, I_> Out(In.nrows(), In.ncols());

        for(I i = 0; i < Out.size(); ++i)
        {
            Out[i] = T_(In[i]);
        }

        return Out;
    }

    static auto Empty() noexcept -> HostMatrix<T_, I_>
    {
        HostMatrix<T_, I_> empty(nullptr, 0, 0);

        return empty;
    }

    static auto Zeros(I nrows, I ncols) -> HostMatrix<T_, I_>
    {
        return HostMatrix<T_, I_>(nrows, ncols);
    }

    static auto Zeros(I dim) -> HostMatrix<T_, I_>
    {
        return HostMatrix<T_, I_>(dim);
    }

    static auto Eye(I nrows, I ncols) -> HostMatrix<T_, I_>
    {
        HostMatrix<T_, I_> Id(nrows, ncols);

        if(!Id.empty())
        {
            I dim = std::min(Id.nrows(), Id.ncols());
            for(I i = 0; i < dim; ++i)
            {
                Id(i, i) = T_(1.);
            }
        }

        return Id;
    }

    static auto Eye(I dim) -> HostMatrix<T_, I_>
    {
        return HostMatrix<T_, I_>::Eye(dim, dim);
    }

    static auto Ones(I nrows, I ncols) -> HostMatrix<T_, I_>
    {
        HostMatrix<T_, I_> ones(nrows, ncols);

        if(!ones.empty())
        {
            for(I i = 0; i < ones.size(); ++i)
            {
                ones[i] = T_(1.);
            }
        }

        return ones;
    }

    static auto Ones(I dim) -> HostMatrix<T_, I_>
    {
        return HostMatrix<T_, I_>::Ones(dim, dim);
    }

    static auto Diag(const HostMatrix<T_, I_>& d)
    {
        if(d.empty() || (std::min(d.nrows(), d.ncols())))
        {
            throw; // TODO: pick correct exception to throw, or return empty matrix
        }

        I dim = std::max(d.nrows(), d.ncols());
        HostMatrix<T_, I_> Z(dim, dim);
        for(I i = 0; i < dim; ++i)
        {
            Z(i, i) = d[i];
        }

        return Z;
    }

    static auto FromList(std::initializer_list<T> list) -> HostMatrix<T_, I_>
    {
        HostMatrix<T_, I_> Z(1, static_cast<I>(list.size()));

        if(!Z.empty())
        {
            I i = 0;
            for(auto iter = list.begin(); iter != list.end(); ++iter)
            {
                Z[i] = static_cast<T>(*iter);
                ++i;
            }
        }

        return Z;
    }

    static auto FromRange(T start_val, T end_val, I num_vals) -> HostMatrix<T_, I_>
    {
        if(num_vals < 1)
        {
            return HostMatrix<T_, I_>::Empty();
        }
        else if(num_vals == 1)
        {
            return HostMatrix<T_, I_>::FromList({start_val});
        }

        I nvals = std::max(num_vals, I(2));
        T t = (end_val - start_val) / static_cast<T>(nvals - 1);

        HostMatrix<T_, I_> Z(1, nvals);
        for(I i = 0; i < Z.size(); ++i)
        {
            Z[i] = start_val + t * static_cast<T>(i);
        }

        return Z;
    }

    HostMatrix(I nrows, I ncols)
        : HostMatrix(nullptr, nrows, ncols)
    {
        if(empty())
        {
            return;
        }

        this->data_ = this->cgc_.alloc(this->size());
        if(this->data_ == nullptr)
        {
            throw std::bad_alloc();
        }
    }

    HostMatrix(I dim)
        : HostMatrix(dim, dim)
    {
    }

    virtual ~HostMatrix() override = default;

    HostMatrix(const HostMatrix& rhs)
    {
        if(this->is_a_wrapper_)
        {
            if(!this->copy_data_from(rhs))
            {
                throw; // TODO: pick adequate exception
            }
        }
        else
        {
            this->nrows_ = rhs.nrows_;
            this->ncols_ = rhs.ncols_;
            this->stride_ = rhs.stride_;
            this->ld_ = rhs.ld_;
            this->empty_ = rhs.empty_;
            this->is_a_wrapper_ = false;
            this->max_size_ = rhs.size();

            this->data_ = this->cgc_.alloc_and_copy(this->size(), rhs.data_);
            if(!this->empty() && (this->data_ == nullptr))
            {
                this->nrows_ = 0;
                this->ncols_ = 0;
                this->ld_ = 0;
                this->empty_ = 0;
                this->max_size_ = 0;
                throw std::bad_alloc();
            }
        }
    }

    HostMatrix& operator=(const HostMatrix& rhs)
    {
        if(this->is_a_wrapper_)
        {
            if(!this->copy_data_from(rhs))
            {
                throw; // TODO: pick adequate exception
            }
        }
        else
        {
            this->nrows_ = rhs.nrows_;
            this->ncols_ = rhs.ncols_;
            this->stride_ = rhs.stride_;
            this->ld_ = rhs.ld_;
            this->empty_ = rhs.empty_;
            this->is_a_wrapper_ = false;
            this->max_size_ = rhs.size();

            this->data_ = this->cgc_.alloc_and_copy(this->size(), rhs.data_);
            if(!this->empty() && (this->data_ == nullptr))
            {
                this->nrows_ = 0;
                this->ncols_ = 0;
                this->ld_ = 0;
                this->empty_ = 0;
                this->max_size_ = 0;
                throw std::bad_alloc();
            }
        }

        return *this;
    }

    // TODO: handle wrappers properly
    HostMatrix(HostMatrix&&) = default;

    // TODO: handle wrappers properly
    HostMatrix& operator=(HostMatrix&&) = default;

    virtual T const* data() const override
    {
        return data_;
    }

    virtual T* data() override
    {
        return data_;
    }

    [[maybe_unused]] virtual T* copy_to(T* dest) const override
    {
        auto mptr = memmove(dest, data(), num_bytes());

        return static_cast<T*>(mptr);
    }

    [[maybe_unused]] virtual bool copy_to(std::vector<T>& dest) const override
    {
        dest.resize(this->size());
        [[maybe_unused]] auto volatile mptr
            = copy_to(dest.data()); //memcpy(dest.data(), this->data_, this->num_bytes());

        return true;
    }

    [[maybe_unused]] virtual bool copy_data_from(const MatrixInterface<T_, I_>& src) override
    {
        if((nrows() < src.nrows()) || (ncols() < src.ncols()))
        {
            return false;
        }

        /* [[maybe_unused]] auto volatile mptr */
        /* = memmove(this->data(), src.data(), src.num_bytes()); */
        for(I j = 0; j < src.ncols(); ++j)
        {
            for(I i = 0; i < src.nrows(); ++i)
            {
                this->operator()(i, j) = src(i, j);
            }
        }

        return true;
    }

    [[maybe_unused]] virtual bool set_data_from(const MatrixInterface<T_, I_>& src) override
    {
        if(empty() || (max_size_ < src.size()))
        {
            return false;
        }

        [[maybe_unused]] auto volatile mptr = memmove(this->data(), src.data(), src.num_bytes());
        nrows_ = src.nrows();
        ncols_ = src.ncols();

        return true;
    }

    virtual void set_to_zero() override
    {
        [[maybe_unused]] auto volatile mptr = memset(this->data(), 0, this->num_bytes());
    }

    virtual I nrows() const override
    {
        return nrows_;
    }

    virtual I ncols() const override
    {
        return ncols_;
    }

    virtual I ld() const override
    {
        return ld_;
    }

    virtual I size() const override
    {
        return nrows_ * ncols_;
    }

    virtual I num_bytes() const override
    {
        return sizeof(T) * nrows_ * ncols_;
    }

    virtual bool empty() const override
    {
        return empty_;
    }

    // TODO: reshape should return ref to *this
    virtual bool reshape(I nrows, I ncols) override
    {
        if(empty() || (size() != nrows * ncols))
        {
            return false;
        }

        nrows_ = nrows;
        ncols_ = ncols;
        ld_ = nrows;

        return true;
    }

    virtual T operator()(I i, I j) const override
    {
        return data_[i + static_cast<std::int64_t>(ld_) * j];
    }

    virtual T& operator()(I i, I j) override
    {
        return data_[i + static_cast<std::int64_t>(ld_) * j];
    }

    virtual T operator[](I i) const override
    {
        return data_[i];
    }

    virtual T& operator[](I i) override
    {
        return data_[i];
    }

    virtual S max_coeff_norm() const override
    {
        S norm = S(0.);
        for(I i = 0; i < size(); ++i)
        {
            S el = detail::abs(this->operator[](i));
            norm = (norm > el) ? norm : el;
        }

        return norm;
    }

    virtual S max_col_norm() const override
    {
        S norm = S(0.);
        auto col_norm = HostMatrix<S, I_>::Zeros(1, ncols());

        for(I j = 0; j < ncols(); ++j)
        {
            for(I i = 0; i < nrows(); ++i)
            {
                col_norm(0, j) += detail::norm(this->operator()(i, j));
            }
        }

        norm = col_norm.max_coeff_norm();
        return std::sqrt(norm);
    }

    virtual S norm() const override
    {
        S norm = S(0.);
        auto col_norm = HostMatrix<S, I_>::Zeros(1, ncols());

        for(I j = 0; j < ncols(); ++j)
        {
            for(I i = 0; i < nrows(); ++i)
            {
                col_norm(0, j) += detail::norm(this->operator()(i, j));
            }
        }

        for(I i = 0; i < col_norm.size(); ++i)
        {
            norm += col_norm[i];
        }
        return std::sqrt(norm);
    }

    virtual HostMatrix<T_, I_> row(I k) const
    {
        if(empty() || (k < 0) || (k >= nrows()))
        {
            return HostMatrix<T_, I_>::Empty();
        }

        HostMatrix<T_, I_> out(1, ncols());

        for(I i = 0; i < ncols(); ++i)
        {
            out[i] = this->operator()(k, i);
        }

        return out;
    }

    virtual HostMatrix<T_, I_> col(I k) const
    {
        if(empty() || (k < 0) || (k >= ncols()))
        {
            return HostMatrix<T_, I_>::Empty();
        }

        HostMatrix<T_, I_> out(nrows(), 1);

        for(I i = 0; i < nrows(); ++i)
        {
            out[i] = this->operator()(i, k);
        }

        return out;
    }

    virtual HostMatrix<T_, I_> diag() const
    {
        I dim = std::min(nrows(), ncols());
        HostMatrix<T_, I_> out(1, dim);

        if(!out.empty())
        {
            for(I i = 0; i < dim; ++i)
            {
                out[i] = this->operator()(i, i);
            }
        }

        return out;
    }

    virtual HostMatrix<T_, I_> sub_diag() const
    {
        I dim = std::min(nrows(), ncols()) - 1;
        HostMatrix<T_, I_> out(1, std::max(dim, I(1)));

        if(!out.empty())
        {
            for(I i = 0; i < dim; ++i)
            {
                out[i] = this->operator()(i + 1, i);
            }
        }

        return out;
    }

    virtual HostMatrix<T_, I_> sup_diag() const
    {
        I dim = std::min(nrows(), ncols()) - 1;
        HostMatrix<T_, I_> out(1, dim);

        if(!out.empty())
        {
            for(I i = 0; i < dim; ++i)
            {
                out[i] = this->operator()(i, i + 1);
            }
        }

        return out;
    }

    struct BlockDescriptor
    {
        mutable I from_row_{0};
        mutable I from_col_{0};
        mutable I nrows_{1};
        mutable I ncols_{1};

        BlockDescriptor& from_row(I row)
        {
            from_row_ = row;
            return *this;
        }

        BlockDescriptor& from_col(I col)
        {
            from_col_ = col;
            return *this;
        }

        BlockDescriptor& nrows(I n)
        {
            nrows_ = n;
            return *this;
        }

        BlockDescriptor& ncols(I n)
        {
            ncols_ = n;
            return *this;
        }

        BlockDescriptor& rows_from_to(I from_row, I to_row)
        {
            from_row_ = from_row;
            nrows_ = to_row - from_row + 1;
            return *this;
        }

        BlockDescriptor& cols_from_to(I from_col, I to_col)
        {
            from_col_ = from_col;
            ncols_ = to_col - from_col + 1;
            return *this;
        }

        bool range_check(I max_nrows, I max_ncols) const
        {
            bool in_range = true;

            if((from_row_ < 0) || (from_row_ >= max_nrows))
            {
                in_range = false;
            }
            else if((nrows_ < 1) || (nrows_ > max_nrows))
            {
                in_range = false;
            }
            else if((from_col_ < 0) || (from_col_ >= max_nrows))
            {
                in_range = false;
            }
            else if((ncols_ < 1) || (ncols_ > max_ncols))
            {
                in_range = false;
            }

            return in_range;
        }
    };

    virtual HostMatrix<T_, I_> block(const BlockDescriptor& bd) const
    {
        if(empty() || !bd.range_check(nrows(), ncols()))
        {
            return HostMatrix<T_, I_>::Empty();
        }

        HostMatrix<T_, I_> out(bd.nrows_, bd.ncols_);

        for(I j = 0; j < out.ncols(); ++j)
        {
            for(I i = 0; i < out.nrows(); ++i)
            {
                out(i, j) = this->operator()(i + bd.from_row_, j + bd.from_col_);
            }
        }

        return out;
    }

    virtual HostMatrix<T_, I_>& row(I k, const HostMatrix<T_, I_>& r)
    {
        auto b = BlockDescriptor().from_col(0).ncols(ncols()).from_row(k).nrows(1);
        if(!empty() || b.range_check(nrows(), ncols()))
        {
            for(I i = 0; i < ncols(); ++i)
            {
                this->operator()(k, i) = r[i];
            }
        }

        return *this;
    }

    virtual HostMatrix<T_, I_>& col(I k, const HostMatrix<T_, I_>& c)
    {
        auto b = BlockDescriptor().from_row(0).nrows(nrows()).from_col(k).ncols(1);
        if(!empty() || b.range_check(nrows(), ncols()))
        {
            for(I i = 0; i < nrows(); ++i)
            {
                this->operator()(i, k) = c[i];
            }
        }

        return *this;
    }

    virtual HostMatrix<T_, I_>& diag(const HostMatrix<T_, I_>& d)
    {
        I dim = std::min(nrows(), ncols());
        if(!empty() && (dim == d.size()))
        {
            for(I i = 0; i < dim; ++i)
            {
                this->operator()(i, i) = d[i];
            }
        }

        return *this;
    }

    virtual HostMatrix<T_, I_>& sub_diag(const HostMatrix<T_, I_>& f)
    {
        I dim = std::min(nrows(), ncols()) - 1;

        if(!empty() && (dim == f.size()))
        {
            for(I i = 0; i < dim; ++i)
            {
                this->operator()(i + 1, i) = f[i];
            }
        }

        return *this;
    }

    virtual HostMatrix<T_, I_>& sup_diag(const HostMatrix<T_, I_>& e)
    {
        I dim = std::min(nrows(), ncols()) - 1;

        if(!empty() && (dim == e.size()))
        {
            for(I i = 0; i < dim; ++i)
            {
                this->operator()(i, i + 1) = e[i];
            }
        }

        return *this;
    }

    virtual HostMatrix<T_, I_>& block(const BlockDescriptor& bd, const HostMatrix<T_, I_>& b)
    {
        if(!empty() && bd.range_check(nrows(), ncols()))
        {
            for(I j = 0; j < bd.ncols_; ++j)
            {
                for(I i = 0; i < bd.nrows_; ++i)
                {
                    this->operator()(i + bd.from_row_, j + bd.from_col_) = b(i, j);
                }
            }
        }

        return *this;
    }

    void print() const
    {
        /* for (I i = 0; i < size(); ++i) */
        /* { */
        /*         std::cout << this->data_[i]; */
        /*         if (i != (size() - 1)) */
        /*             std::cout << ", "; */
        /*         else */
        /*             std::cout << "\n" << std::flush; */
        /* } */
        for(I i = 0; i < nrows_; ++i)
        {
            for(I j = 0; j < ncols_; ++j)
            {
                std::cout << this->operator()(i, j);
                if(j != ncols_ - 1)
                    std::cout << ", ";
                else
                    std::cout << "\n" << std::flush;
            }
        }
    }

protected:
    mutable T* data_; ///< memory pointed by `data_` is either managed externally or by `class cgc`
    mutable I nrows_{};
    mutable I ncols_{};
    mutable I stride_{1};
    mutable I ld_{};
    mutable bool empty_{true};
    mutable bool is_a_wrapper_{false};
    mutable I max_size_{};

    HostMatrix(T* in_data, I nrows, I ncols) noexcept
        : data_(in_data)
        , nrows_(nrows)
        , ncols_(ncols)
        , ld_(nrows)
    {
        if((nrows < 1) || (ncols < 1))
        {
            nrows_ = I(0);
            ncols_ = I(0);
            ld_ = I(0);
            empty_ = true;
        }
        else
        {
            empty_ = false;
        }
        is_a_wrapper_ = false;
        max_size_ = nrows_ * ncols_;
    }

    ///
    /// `cgc<T>` wraps calloc() and free() for RAII memory management in C++ code.
    ///
    /// /* template <typename T> */
    class cgc
    {
    public:
        /// Constructor is only defined when type T has a POD (C plain old data) memory layout and is not a pointer
        template <typename U = T, /* dummy type U must depend on T for SFINAE to kick in */
                  typename std::enable_if<std::is_convertible<U, T>::value && std::is_trivial<U>::value
                                          && std::is_standard_layout<U>::value
                                          && !std::is_pointer<U>::value>::type* = nullptr>

        /// Constructor
        cgc() noexcept
        {
        }

        /// Destructor frees all of the remaining allocated memory
        ~cgc()
        {
            clear();
        }

        /// Deleted copy constructor
        cgc(const cgc&) = delete;

        /// Deleted copy assignment
        cgc& operator=(const cgc&) = delete;

        /// Move constructor
        cgc(cgc&& rhs)
        {
            clear();
            ptr_list = std::move(rhs.ptr_list);
        }

        /// Move assignment
        cgc& operator=(cgc&& rhs)
        {
            clear();
            ptr_list = std::move(rhs.ptr_list);

            return *this;
        }

        /// cgc::alloc() returns pointer to n*sizeof(T) (zero initialized) memory block
        T* alloc(std::size_t n = 1)
        {
            if(n == 0)
            {
                return nullptr;
            }

            T* mem_ptr = (T*)calloc(n, sizeof(T));

            if(mem_ptr != nullptr)
            {
                ptr_list.push_back(mem_ptr);
            }

            return mem_ptr;
        }

        /// cgc::alloc_and_copy() returns pointer to n*sizeof(T) memory block with the contents pointed by src
        T* alloc_and_copy(std::size_t n, T* src)
        {
            if((n == 0) || (src == nullptr))
            {
                return nullptr;
            }

            T* mem_ptr = (T*)calloc(n, sizeof(T));

            if(mem_ptr != nullptr)
            {
                ptr_list.push_back(mem_ptr);
                auto num_bytes = n * sizeof(T);
                [[maybe_unused]] auto volatile mptr = memcpy(mem_ptr, src, num_bytes);
            }

            return mem_ptr;
        }

        /// cgc::free() deallocates memory previously allocated by same instance of class and clears ptr
        void free(T*& ptr)
        {
            if(ptr == nullptr)
            {
                return;
            }

            for(auto& p : ptr_list)
            {
                if(p == ptr)
                {
                    ::free(ptr);
                    ptr = nullptr;
                    p = nullptr;
                }
            }
        }

        /// cgc::clear() frees all of the remaining allocated memory
        void clear()
        {
            for(auto& p : ptr_list)
            {
                if(p != nullptr)
                {
                    ::free(p);
                }
            }

            ptr_list.clear();
        }

    private:
        std::vector<T*> ptr_list{};
    } cgc_{};
};

template <template <typename, typename> class HostMatrix_, typename T, typename I>
auto operator+(const HostMatrix_<T, I>& A, const HostMatrix_<T, I>& B)
{
    if((A.nrows() != B.nrows()) || (A.ncols() != B.ncols()))
    {
        std::string error_message = std::string("Error computing A + B: ") + "(A.nrows() [= "
            + std::to_string(A.nrows()) + "] " + "!= B.nrows()) [= " + std::to_string(B.nrows())
            + "] " + "|| (A.ncols() [= " + std::to_string(A.ncols()) + "] "
            + "!= B.ncols()) [= " + std::to_string(B.ncols()) + ")";

        throw std::domain_error(error_message);
    }

    I nrows = A.nrows();
    I ncols = A.ncols();
    I size = A.size();
    HostMatrix_<T, I> Z(nrows, ncols);

    for(I i = 0; i < size; ++i)
    {
        Z[i] = A[i] + B[i];
    }

    return Z;
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
auto operator-(const HostMatrix_<T, I>& A, const HostMatrix_<T, I>& B)
{
    if((A.nrows() != B.nrows()) || (A.ncols() != B.ncols()))
    {
        std::string error_message = std::string("Error computing A - B: ") + "(A.nrows() [= "
            + std::to_string(A.nrows()) + "] " + "!= B.nrows()) [= " + std::to_string(B.nrows())
            + "] " + "|| (A.ncols() [= " + std::to_string(A.ncols()) + "] "
            + "!= B.ncols()) [= " + std::to_string(B.ncols()) + ")";

        throw std::domain_error(error_message);
    }

    I nrows = A.nrows();
    I ncols = A.ncols();
    I size = A.size();
    HostMatrix_<T, I> Z(nrows, ncols);

    for(I i = 0; i < size; ++i)
    {
        Z[i] = A[i] - B[i];
    }

    return Z;
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
auto operator+(const HostMatrix_<T, I>& A)
{
    return A;
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
auto operator-(const HostMatrix_<T, I>& A)
{
    I nrows = A.nrows();
    I ncols = A.ncols();
    I size = A.size();
    HostMatrix_<T, I> Z(nrows, ncols);

    for(I i = 0; i < size; ++i)
    {
        Z[i] = -A[i];
    }

    return Z;
}

template <template <typename, typename> class HostMatrix_, typename T, typename I, typename S>
auto operator*(const HostMatrix_<T, I>& A, S alpha)
{
    I nrows = A.nrows();
    I ncols = A.ncols();
    HostMatrix_<T, I> Z(nrows, ncols);

    for(I j = 0; j < ncols; ++j)
    {
        for(I i = 0; i < nrows; ++i)
        {
            Z(i, j) = A(i, j) * static_cast<T>(alpha);
        }
    }

    return Z;
}

template <template <typename, typename> class HostMatrix_, typename T, typename I, typename S>
auto operator*(S alpha, const HostMatrix_<T, I>& A)
{
    I nrows = A.nrows();
    I ncols = A.ncols();
    HostMatrix_<T, I> Z(nrows, ncols);

    for(I j = 0; j < ncols; ++j)
    {
        for(I i = 0; i < nrows; ++i)
        {
            Z(i, j) = static_cast<T>(alpha) * A(i, j);
        }
    }

    return Z;
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
auto operator*(const HostMatrix_<T, I>& A, const HostMatrix_<T, I>& B)
{
    if((A.ncols() != B.nrows()))
    {
        std::string error_message = "Error computing A * B: A.ncols() [= "
            + std::to_string(A.ncols()) + "] != B.nrows() [= " + std::to_string(B.nrows()) + "]";

        throw std::domain_error(error_message);
    }

    I nrows = A.nrows();
    I ncols = B.ncols();
    HostMatrix_<T, I> Z(nrows, ncols);

    if constexpr(std::is_same<std::decay_t<I>, int>::value)
    {
        detail::lapack_gemm(A.data(), A.nrows(), A.ncols(), B.data(), B.ncols(), Z.data());
    }
    else
    {
        bool within_lapack_limits = (static_cast<int64_t>(A.nrows())
                                     <= static_cast<int64_t>(std::numeric_limits<int>::max()))
            && (static_cast<int64_t>(A.ncols())
                <= static_cast<int64_t>(std::numeric_limits<int>::max()))
            && (static_cast<int64_t>(B.ncols())
                <= static_cast<int64_t>(std::numeric_limits<int>::max()));

        if(within_lapack_limits)
        {
            detail::lapack_gemm(A.data(), A.nrows(), A.ncols(), B.data(), B.ncols(), Z.data());
        }
        else
        {
            I dim = A.ncols();
            for(I j = 0; j < ncols; ++j)
                for(I i = 0; i < nrows; ++i)
                    for(I k = 0; k < dim; ++k)
                        Z(i, j) += A(i, k) * B(k, j);
        }
    }

    return Z;
}

template <template <typename, typename> class HostMatrix_, typename T, typename I, typename S>
auto operator/(const HostMatrix_<T, I>& A, S alpha)
{
    I nrows = A.nrows();
    I ncols = A.ncols();
    HostMatrix_<T, I> Z(nrows, ncols);

    for(I j = 0; j < ncols; ++j)
    {
        for(I i = 0; i < nrows; ++i)
        {
            Z(i, j) = A(i, j) / static_cast<T>(alpha);
        }
    }

    return Z;
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
auto cat(const HostMatrix_<T, I>& A, const HostMatrix_<T, I>& B)
{
    I nrows{}, ncols{};

    if((A.empty()) && ((B.nrows() == 1) || (B.ncols() == 1)))
    {
        return B;
    }
    else if((B.empty()) && ((A.nrows() == 1) || (A.ncols() == 1)))
    {
        return A;
    }
    else if((A.nrows() == B.nrows()) && (A.nrows() == I(1)))
    {
        nrows = A.nrows();
        ncols = A.ncols() + B.ncols();
    }
    else if((A.ncols() == B.ncols()) && (A.ncols() == I(1)))
    {
        nrows = A.nrows() + B.nrows();
        ncols = A.ncols();
    }
    else
    {
        throw std::domain_error(
            "Error computing cat(A, B): A, B must both be row or column vectors");
    }
    HostMatrix_<T, I> Z(nrows, ncols);

    for(I i = 0; i < A.size(); ++i)
    {
        Z[i] = A[i];
    }

    I a_size = A.size();
    for(I i = 0; i < B.size(); ++i)
    {
        Z[i + a_size] = B[i];
    }

    return Z;
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
HostMatrix_<T, I> transpose(const HostMatrix_<T, I>& A)
{
    HostMatrix_<T, I> Z(A.ncols(), A.nrows());

    for(I j = 0; j < Z.ncols(); ++j)
    {
        for(I i = 0; i < Z.nrows(); ++i)
        {
            Z(i, j) = A(j, i);
        }
    }
    return Z;
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
HostMatrix_<T, I> conjugate(const HostMatrix_<T, I>& A)
{
    HostMatrix_<T, I> Z(A.nrows(), A.ncols());

    // TODO: avoid for loop when `T` is real
    for(I i = 0; i < A.size(); ++i)
    {
        Z[i] = detail::conj(A[i]);
    }

    return Z;
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
HostMatrix_<T, I> adjoint(const HostMatrix_<T, I>& A)
{
    HostMatrix_<T, I> Z(A.nrows(), A.ncols());

    Z = conjugate(A);
    Z = transpose(Z);

    return Z;
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
auto qr(const HostMatrix_<T, I>& A)
    -> std::tuple<HostMatrix_<T, I> /* Q */, HostMatrix_<T, I> /* R */>
{
    I nrows = A.nrows();
    I ncols = A.ncols();
    HostMatrix_<T, I> Q(nrows, ncols), R(ncols, ncols);

    if constexpr(std::is_same<std::decay_t<I>, int>::value)
    {
        detail::lapack_qr(A.data(), nrows, ncols, Q.data(), R.data());
    }
    else
    {
        bool within_lapack_limits
            = (static_cast<int64_t>(nrows) <= static_cast<int64_t>(std::numeric_limits<int>::max()))
            && (static_cast<int64_t>(ncols) <= static_cast<int64_t>(std::numeric_limits<int>::max()));

        if(within_lapack_limits)
        {
            detail::lapack_qr(A.data(), static_cast<int>(nrows), static_cast<int>(ncols), Q.data(),
                              R.data());
        }
        else
        {
            throw std::domain_error("Error computing qr(A): A.nrows() && A.ncols() must be smaller "
                                    "or equal to INT_MAX");
        }
    }

    return std::make_tuple(Q, R);
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
auto eig_upper(const HostMatrix_<T, I>& A)
    -> std::tuple<HostMatrix_<T, I> /* Eigenvectors: U */, HostMatrix_<T, I> /* Eigenvalues: Lambda */>
{
    I nrows = A.nrows();
    I ncols = A.ncols();
    if(nrows != ncols)
    {
        std::string error_message = "Error computing eig(A): A.nrows() [= "
            + std::to_string(A.nrows()) + "] != A.ncols() [= " + std::to_string(A.ncols()) + "]";

        throw std::domain_error(error_message);
    }

    I dim = nrows;
    HostMatrix_<T, I> U(dim), Lambda(1, dim);

    if constexpr(std::is_same<std::decay_t<I>, int>::value)
    {
        detail::lapack_sym_eig_upper(A.data(), dim, U.data(), Lambda.data());
    }
    else
    {
        bool within_lapack_limits
            = static_cast<int64_t>(dim) <= static_cast<int64_t>(std::numeric_limits<int>::max());

        if(within_lapack_limits)
        {
            detail::lapack_sym_eig_upper(A.data(), static_cast<int>(dim), U.data(), Lambda.data());
        }
        else
        {
            throw std::domain_error("Error computing eig(A): A.nrows() && A.ncols() must be "
                                    "smaller or equal to INT_MAX");
        }
    }

    return std::make_tuple(U, Lambda);
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
auto eig_lower(const HostMatrix_<T, I>& A)
    -> std::tuple<HostMatrix_<T, I> /* Eigenvectors: U */, HostMatrix_<T, I> /* Eigenvalues: Lambda */>
{
    I nrows = A.nrows();
    I ncols = A.ncols();
    if(nrows != ncols)
    {
        throw std::domain_error("Error computing eig(A): A.nrows() != A.ncols()");
    }

    I dim = nrows;
    HostMatrix_<T, I> U(dim), Lambda(1, dim);

    if constexpr(std::is_same<std::decay_t<I>, int>::value)
    {
        detail::lapack_sym_eig_upper(A.data(), dim, U.data(), Lambda.data());
    }
    else
    {
        bool within_lapack_limits
            = static_cast<int64_t>(dim) <= static_cast<int64_t>(std::numeric_limits<int>::max());

        if(within_lapack_limits)
        {
            detail::lapack_sym_eig_lower(A.data(), static_cast<int>(dim), U.data(), Lambda.data());
        }
        else
        {
            throw std::domain_error("Error computing eig(A): A.nrows() && A.ncols() must be "
                                    "smaller or equal to INT_MAX");
        }
    }

    return std::make_tuple(U, Lambda);
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
auto svd(const HostMatrix_<T, I>& A) -> std::tuple<HostMatrix_<T, I> /* Left Singular Vectors: U */,
                                                   HostMatrix_<T, I> /* Singular Values */,
                                                   HostMatrix_<T, I> /* Right Singular Vectors */>
{
    using S = typename HostMatrix_<T, I>::S;

    I nrows = A.nrows();
    I ncols = A.ncols();

    I dim = std::min(nrows, ncols);
    HostMatrix_<T, I> U(nrows, nrows), V(ncols, ncols), Sigma(nrows, ncols);
    HostMatrix_<S, I> sigma_diag(dim, 1);

    if constexpr(std::is_same<std::decay_t<I>, int>::value)
    {
        detail::lapack_ge_svd(A.data(), nrows, ncols, U.data(), sigma_diag.data(), V.data());
    }
    else
    {
        bool within_lapack_limits
            = static_cast<std::int64_t>(nrows) * static_cast<std::int64_t>(ncols)
                <= static_cast<std::int64_t>(std::numeric_limits<int>::max())
            && static_cast<std::int64_t>(nrows)
                <= static_cast<std::int64_t>(std::numeric_limits<int>::max())
            && static_cast<std::int64_t>(ncols)
                <= static_cast<std::int64_t>(std::numeric_limits<int>::max());

        if(within_lapack_limits)
        {
            detail::lapack_ge_svd(A.data(), static_cast<int>(nrows), static_cast<int>(ncols),
                                  U.data(), sigma_diag.data(), V.data());
        }
        else
        {
            throw std::domain_error(
                "Error computing svd(A): A.nrows(), A.ncols(), A.nrows()*A.ncols() must be "
                "smaller or equal to INT_MAX");
        }
    }

    // Lapack *gesvd returns V^* instead of V.
    Sigma.diag(HostMatrix_<T, I>::Convert(sigma_diag));
    V = adjoint(V);
    return std::make_tuple(U, Sigma, V);
}

template <template <typename, typename> class HostMatrix_, typename T, typename I>
auto inv(const HostMatrix_<T, I>& A) -> HostMatrix_<T, I> /* Pseudo-Inverse of A */
{
    using S = typename HostMatrix_<T, I>::S;

    auto [U, Sigma, V] = svd(A);
    I nrows = A.nrows();
    I ncols = A.ncols();
    I dim = std::min(nrows, ncols);

    for(I i = 0; i < dim; ++i)
    {
        if(std::abs(Sigma(i, i)) > std::max(std::numeric_limits<S>::min(), S(0)))
        {
            Sigma(i, i) = T(1) / Sigma(i, i);
        }
        else
        {
            Sigma(i, i) = T(0);
        }
    }

    auto iA = adjoint(U * Sigma * adjoint(V));
    return iA;
}

} // namespace matxu
