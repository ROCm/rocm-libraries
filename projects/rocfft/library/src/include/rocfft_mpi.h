// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCFFT_MPI_H
#define ROCFFT_MPI_H

#ifdef ROCFFT_MPI_ENABLE
#include <mpi.h>
#if defined(OPEN_MPI) && OPEN_MPI
#include <mpi-ext.h>
#endif

#include "../../../shared/array_predicate.h"
#include "rocfft/rocfft.h"

class MPI_Comm_wrapper_t
{
public:
    MPI_Comm_wrapper_t() = default;

    static MPI_Comm_wrapper_t from_raw(MPI_Comm raw_comm)
    {
        MPI_Comm_wrapper_t wrap;
        wrap.mpi_comm = raw_comm;
        return wrap;
    }

    // conversion to unwrapped communicator for passing to MPI APIs
    operator MPI_Comm() const
    {
        return mpi_comm;
    }

    MPI_Comm_wrapper_t(const MPI_Comm_wrapper_t&) = delete;
    MPI_Comm_wrapper_t& operator=(const MPI_Comm_wrapper_t&) = delete;

    // move communicator
    MPI_Comm_wrapper_t(MPI_Comm_wrapper_t&& other)
    {
        std::swap(this->mpi_comm, other.mpi_comm);
    }
    MPI_Comm_wrapper_t& operator=(MPI_Comm_wrapper_t&& other)
    {
        std::swap(this->mpi_comm, other.mpi_comm);
        return *this;
    }

    ~MPI_Comm_wrapper_t()
    {
        free();
    }

    void free()
    {
        if(mpi_comm != MPI_COMM_NULL)
            MPI_Comm_free(&mpi_comm);
        mpi_comm = MPI_COMM_NULL;
    }

    void duplicate(MPI_Comm in_comm)
    {
        free();
        if(in_comm != MPI_COMM_NULL && MPI_Comm_dup(in_comm, &mpi_comm) != MPI_SUCCESS)
        {
            throw std::runtime_error("failed to duplicate MPI communicator");
        }
    }

    // check if communicator has been initialized
    operator bool() const
    {
        return mpi_comm != MPI_COMM_NULL;
    }
    bool operator!() const
    {
        return mpi_comm == MPI_COMM_NULL;
    }

private:
    MPI_Comm mpi_comm = MPI_COMM_NULL;
};

// RAII wrapper around MPI_Datatypes
class MPI_Datatype_vector_wrapper_t
{
public:
    MPI_Datatype_vector_wrapper_t(size_t size_bytes)
        : type(MPI_DATATYPE_NULL)
    {
        auto rcmpi = MPI_Type_contiguous(size_bytes, MPI_BYTE, &type);
        if(rcmpi != MPI_SUCCESS)
            throw std::runtime_error("MPI_Type_contiguous failed with code: "
                                     + std::to_string(rcmpi));
        rcmpi = MPI_Type_commit(&type);
        if(rcmpi != MPI_SUCCESS)
            throw std::runtime_error("MPI_Type_commit failed with code: " + std::to_string(rcmpi));
    }
    ~MPI_Datatype_vector_wrapper_t()
    {
        MPI_Type_free(&type);
    }

    // Convert to unwrapped type
    operator MPI_Datatype() const
    {
        return type;
    }

private:
    MPI_Datatype type;
};

// Helper function to get the MPI data type from the underlying type of the variable.
template <class Tval>
inline MPI_Datatype type_to_mpi_type()
{
    if(std::is_same<Tval, char>::value)
    {
        return MPI_CHAR;
    }
    else if(std::is_same<Tval, signed char>::value)
    {
        return MPI_SIGNED_CHAR;
    }
    else if(std::is_same<Tval, unsigned char>::value)
    {
        return MPI_UNSIGNED_CHAR;
    }
    else if(std::is_same<Tval, short>::value)
    {
        return MPI_SHORT;
    }
    else if(std::is_same<Tval, int>::value)
    {
        return MPI_INT;
    }
    else if(std::is_same<Tval, unsigned int>::value)
    {
        return MPI_UNSIGNED;
    }
    else if(std::is_same<Tval, long int>::value)
    {
        return MPI_LONG;
    }
    else if(std::is_same<Tval, unsigned long int>::value)
    {
        return MPI_UNSIGNED_LONG;
    }
    else if(std::is_same<Tval, long long int>::value)
    {
        return MPI_LONG_LONG;
    }
    else if(std::is_same<Tval, unsigned long long int>::value)
    {
        return MPI_UNSIGNED_LONG_LONG;
    }
    else if(std::is_same<Tval, float>::value)
    {
        return MPI_FLOAT;
    }
    else if(std::is_same<Tval, double>::value)
    {
        return MPI_DOUBLE;
    }
    else if(std::is_same<Tval, long double>::value)
    {
        return MPI_LONG_DOUBLE;
    }
    else if(std::is_same<Tval, int8_t>::value)
    {
        return MPI_INT8_T;
    }
    else if(std::is_same<Tval, int16_t>::value)
    {
        return MPI_INT16_T;
    }
    else if(std::is_same<Tval, int32_t>::value)
    {
        return MPI_INT32_T;
    }
    else if(std::is_same<Tval, int64_t>::value)
    {
        return MPI_INT64_T;
    }
    else if(std::is_same<Tval, uint8_t>::value)
    {
        return MPI_UINT8_T;
    }
    else if(std::is_same<Tval, uint16_t>::value)
    {
        return MPI_UINT16_T;
    }
    else if(std::is_same<Tval, uint32_t>::value)
    {
        return MPI_UINT32_T;
    }
    else if(std::is_same<Tval, uint64_t>::value)
    {
        return MPI_UINT64_T;
    }
    else if(std::is_same<Tval, rocfft_complex<float>>::value)
    {
        return MPI_C_FLOAT_COMPLEX;
    }
    else if(std::is_same<Tval, rocfft_complex<double>>::value)
    {
        return MPI_C_DOUBLE_COMPLEX;
    }
    else if(std::is_same<Tval, rocfft_complex<long double>>::value)
    {
        return MPI_C_LONG_DOUBLE_COMPLEX;
    }
    else if(std::is_same<Tval, rocfft_fp16>::value)
    {
        static MPI_Datatype_vector_wrapper_t ROCFFT_MPI_HALF{sizeof(Tval)};
        return ROCFFT_MPI_HALF;
    }
    else if(std::is_same<Tval, rocfft_complex<rocfft_fp16>>::value)
    {
        static MPI_Datatype_vector_wrapper_t ROCFFT_MPI_COMPLEX_HALF{sizeof(Tval)};
        return ROCFFT_MPI_COMPLEX_HALF;
    }

    // We did not find the data type: return a null data type.
    return MPI_DATATYPE_NULL;
}

// return an MPI data type suitable for elements in a rocFFT array
inline MPI_Datatype rocfft_type_to_mpi_type(rocfft_precision precision, rocfft_array_type arrayType)
{
    switch(precision)
    {
    case rocfft_precision_half:
        return array_type_is_complex(arrayType) ? type_to_mpi_type<rocfft_complex<rocfft_fp16>>()
                                                : type_to_mpi_type<rocfft_fp16>();
    case rocfft_precision_single:
        return array_type_is_complex(arrayType) ? type_to_mpi_type<rocfft_complex<float>>()
                                                : type_to_mpi_type<float>();
    case rocfft_precision_double:
        return array_type_is_complex(arrayType) ? type_to_mpi_type<rocfft_complex<double>>()
                                                : type_to_mpi_type<double>();
    }
}

inline MPI_Comm_wrapper_t make_subcommunicator(MPI_Comm parent_comm, const std::vector<int>& ranks)
{
    if(ranks.empty())
        return MPI_Comm_wrapper_t{};

    MPI_Group parent_group = MPI_GROUP_NULL, sub_group = MPI_GROUP_NULL;
    MPI_Comm  new_comm = MPI_COMM_NULL;

    auto rcmpi = MPI_Comm_group(parent_comm, &parent_group);
    if(rcmpi != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_group failed with code: " + std::to_string(rcmpi));

    rcmpi = MPI_Group_incl(parent_group, static_cast<int>(ranks.size()), ranks.data(), &sub_group);
    MPI_Group_free(&parent_group);
    if(rcmpi != MPI_SUCCESS)
    {
        if(sub_group != MPI_GROUP_NULL)
            MPI_Group_free(&sub_group);
        throw std::runtime_error("MPI_Group_incl failed with code: " + std::to_string(rcmpi));
    }

    rcmpi = MPI_Comm_create(parent_comm, sub_group, &new_comm);
    MPI_Group_free(&sub_group);
    if(rcmpi != MPI_SUCCESS)
        throw std::runtime_error("MPI_Comm_create failed with code: " + std::to_string(rcmpi));

    if(new_comm == MPI_COMM_NULL)
        return MPI_Comm_wrapper_t{};

    return MPI_Comm_wrapper_t::from_raw(new_comm);
}

#else

typedef int MPI_Comm;
class MPI_Comm_wrapper_t
{
public:
    MPI_Comm_wrapper_t() {}
    static MPI_Comm_wrapper_t from_raw(MPI_Comm)
    {
        return MPI_Comm_wrapper_t{};
    }
    // allow conversion to bool (always false)
    operator bool() const
    {
        return false;
    }
};

#endif

#endif
