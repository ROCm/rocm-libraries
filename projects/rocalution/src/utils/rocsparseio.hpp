/* ************************************************************************
 * Copyright (C) 2024 Advanced Micro Devices, Inc.
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

#pragma once

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

//
//
//
#define ROCSPARSEIO_CHECK(ROCSPARSEIO_CHECK_todo)                                               \
    {                                                                                           \
        rocsparseio::status_t check_status = (ROCSPARSEIO_CHECK_todo);                          \
        if(check_status != rocsparseio::status_t::success)                                      \
        {                                                                                       \
            fprintf(stderr, "ROCSPARSEIO_CHECK FAILED, LINE %d FILE %s\n", __LINE__, __FILE__); \
            return check_status;                                                                \
        }                                                                                       \
    }                                                                                           \
    (void)0

//
//
//
#define ROCSPARSEIO_CHECK_ARG(ROCSPARSEIO_CHECK_ARG_cond, ROCSPARSEIO_CHECK_ARG_status) \
    if(ROCSPARSEIO_CHECK_ARG_cond)                                                      \
    {                                                                                   \
        return ROCSPARSEIO_CHECK_ARG_status;                                            \
    }                                                                                   \
    (void)0

//
//
//
#define ROCSPARSEIO_C_CHECK(ROCSPARSEIO_C_CHECK_todo)                    \
    {                                                                    \
        rocsparseio::status_t check_status = (ROCSPARSEIO_C_CHECK_todo); \
        if(check_status != rocsparseio::status_t::success)               \
        {                                                                \
            fprintf(stderr, "ROCSPARSEIO_CHECK FAILED\n");               \
            return (rocsparseio_status)check_status;                     \
        }                                                                \
    }                                                                    \
    (void)0

//
//
//
#define ROCSPARSEIO_C_CHECK_ARG(ROCSPARSEIO_C_CHECK_ARG_cond, ROCSPARSEIO_C_CHECK_ARG_status) \
    if(ROCSPARSEIO_C_CHECK_ARG_cond)                                                          \
    {                                                                                         \
        return (rocsparseio_status)ROCSPARSEIO_C_CHECK_ARG_status;                            \
    }

#include "rocsparseio.h"

namespace rocsparseio
{

#define ROCSPARSEIO_ENUM_EACH_STATUS                \
    ROCSPARSEIO_ENUM_STATUS(success)                \
    ROCSPARSEIO_ENUM_STATUS(invalid_handle)         \
    ROCSPARSEIO_ENUM_STATUS(invalid_pointer)        \
    ROCSPARSEIO_ENUM_STATUS(invalid_value)          \
    ROCSPARSEIO_ENUM_STATUS(invalid_enum)           \
    ROCSPARSEIO_ENUM_STATUS(invalid_file)           \
    ROCSPARSEIO_ENUM_STATUS(invalid_file_operation) \
    ROCSPARSEIO_ENUM_STATUS(invalid_format)         \
    ROCSPARSEIO_ENUM_STATUS(invalid_mode)           \
    ROCSPARSEIO_ENUM_STATUS(invalid_size)           \
    ROCSPARSEIO_ENUM_STATUS(invalid_memory)

    //!
    //! @brief c++11 struct for status enum.
    //!
    struct status_t
    {

#define ROCSPARSEIO_ENUM_STATUS(ROCSPARSEIO_ENUM_STATUS_x) \
    ROCSPARSEIO_ENUM_STATUS_x = rocsparseio_status_##ROCSPARSEIO_ENUM_STATUS_x,

        typedef enum value_type_ : rocsparseio_status_t
        {
            ROCSPARSEIO_ENUM_EACH_STATUS
        } value_type;

#undef ROCSPARSEIO_ENUM_STATUS

        value_type       value{};
        inline constexpr operator value_type() const
        {
            return this->value;
        }
        inline explicit constexpr operator rocsparseio_status() const
        {
            return (rocsparseio_status)this->value;
        }

        inline constexpr status_t() {}
        inline constexpr status_t(rocsparseio_status_t ival)
            : value((value_type)ival)
        {
        }

        inline constexpr status_t(rocsparseio_status ival)
            : value((value_type)ival)
        {
        }

        inline bool is_invalid() const
        {
#define ROCSPARSEIO_ENUM_STATUS(x_) case x_:
            switch(this->value)
            {
                ROCSPARSEIO_ENUM_EACH_STATUS
                {
                    return false;
                }
            }
#undef ROCSPARSEIO_ENUM_STATUS
            return true;
        }

        inline const char* to_string() const
        {
#define ROCSPARSEIO_ENUM_STATUS(x_) \
    case x_:                        \
    {                               \
        return #x_;                 \
    }
            switch(this->value)
            {
                ROCSPARSEIO_ENUM_EACH_STATUS;
            }
#undef ROCSPARSEIO_ENUM_STATUS
            return "rocsparseio-unknown";
        }

#undef ROCSPARSEIO_ENUM_EACH_STATUS
    };

    //!
    //! @brief c++11 struct for format enum.
    //!
    struct format_t
    {
        typedef enum value_type_ : rocsparseio_format_t
        {
            dense_vector = rocsparseio_format_dense_vector,
            dense_matrix = rocsparseio_format_dense_matrix,
            sparse_csx   = rocsparseio_format_sparse_csx,
            sparse_gebsx = rocsparseio_format_sparse_gebsx,
            sparse_coo   = rocsparseio_format_sparse_coo,
            sparse_ell   = rocsparseio_format_sparse_ell,
            sparse_hyb   = rocsparseio_format_sparse_hyb,
            sparse_dia   = rocsparseio_format_sparse_dia,
            sparse_mcsx  = rocsparseio_format_sparse_mcsx,
        } value_type;

        value_type       value{};
        inline constexpr operator value_type() const
        {
            return this->value;
        }
        inline constexpr format_t(){};
        inline constexpr format_t(rocsparseio_format_t ival_)
            : value((value_type)ival_)
        {
        }

        inline constexpr format_t(rocsparseio_format ival_)
            : value((value_type)ival_)
        {
        }

        inline explicit constexpr operator rocsparseio_format() const
        {
            return (rocsparseio_format)this->value;
        }

        inline bool is_invalid() const
        {
            switch(this->value)
            {
            case dense_vector:
            case dense_matrix:
            case sparse_csx:
            case sparse_gebsx:
            case sparse_coo:
            case sparse_ell:
            case sparse_dia:
            case sparse_hyb:
            case sparse_mcsx:
            {
                return false;
            }
            }
            return true;
        }

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(dense_matrix);
                CASE(dense_vector);
                CASE(sparse_csx);
                CASE(sparse_mcsx);
                CASE(sparse_gebsx);
                CASE(sparse_coo);
                CASE(sparse_ell);
                CASE(sparse_dia);
                CASE(sparse_hyb);
#undef CASE
            }
            return "unknown";
        }
    };

    //!
    //! @brief c++11 struct for order enum.
    //!
    struct order_t
    {
        typedef enum value_type_ : rocsparseio_order_t
        {
            row    = rocsparseio_order_row,
            column = rocsparseio_order_column,
        } value_type;

        value_type       value{};
        inline constexpr operator value_type() const
        {
            return this->value;
        }
        inline explicit constexpr operator rocsparseio_order() const
        {
            return (rocsparseio_order)this->value;
        }

        //    order_t(){};
        inline constexpr order_t() {}
        inline constexpr order_t(rocsparseio_order_t ival_)
            : value((value_type)ival_)
        {
        }

        inline bool is_invalid() const
        {
            switch(this->value)
            {
            case row:
            case column:
            {
                return false;
            }
            }
            return true;
        }

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(row);
                CASE(column);
#undef CASE
            }
            return "unknown";
        }
    };

    //!
    //! @brief c++11 struct for direction enum.
    //!
    struct direction_t
    {
        typedef enum value_type_ : rocsparseio_direction_t
        {
            row    = rocsparseio_direction_row,
            column = rocsparseio_direction_column,
        } value_type;

        value_type       value{};
        inline constexpr operator value_type() const
        {
            return this->value;
        }
        inline explicit constexpr operator rocsparseio_direction() const
        {
            return (rocsparseio_direction)this->value;
        }
        inline constexpr direction_t(){};
        inline constexpr direction_t(rocsparseio_direction_t ival_)
            : value((value_type)ival_)
        {
        }

        inline bool is_invalid() const
        {
            switch(this->value)
            {
            case row:
            case column:
            {
                return false;
            }
            }
            return true;
        }

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(row);
                CASE(column);
#undef CASE
            }
            return "unknown";
        }
    };

    //!
    //! @brief c++11 struct for type enum.
    //!
    struct type_t
    {
        typedef enum value_type_ : rocsparseio_type_t
        {
            int32     = rocsparseio_type_int32,
            int64     = rocsparseio_type_int64,
            float32   = rocsparseio_type_float32,
            float64   = rocsparseio_type_float64,
            complex32 = rocsparseio_type_complex32,
            complex64 = rocsparseio_type_complex64,
        } value_type;

        value_type       value{};
        inline constexpr operator value_type() const
        {
            return this->value;
        };
        inline explicit constexpr operator rocsparseio_type() const
        {
            return (rocsparseio_type)this->value;
        };
        constexpr type_t(){};
        constexpr type_t(rocsparseio_type_t ival_)
            : value((value_type)ival_){};

        template <typename T>
        static constexpr type_t convert();

        inline uint64_t size() const
        {
            switch(this->value)
            {
            case int32:
            {
                return sizeof(int32_t);
            }
            case int64:
            {
                return sizeof(int64_t);
            }
            case float32:
            {
                return sizeof(float);
            }
            case float64:
            {
                return sizeof(double);
            }
            case complex32:
            {
                return sizeof(float) * 2;
            }
            case complex64:
            {
                return sizeof(double) * 2;
            }
            }
            return 0;
        };

        inline bool is_invalid() const
        {
            switch(this->value)
            {
            case int32:
            case int64:
            case float32:
            case float64:
            case complex32:
            case complex64:
            {
                return false;
            }
            }
            return true;
        };

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(int32);
                CASE(int64);
                CASE(float32);
                CASE(float64);
                CASE(complex32);
                CASE(complex64);
#undef CASE
            }
            return "unknown";
        }
    };

    struct index_base_t
    {
        typedef enum value_index_base_ : rocsparseio_index_base_t
        {
            zero = rocsparseio_index_base_zero,
            one  = rocsparseio_index_base_one
        } value_index_base;

        value_index_base value{};
        inline constexpr operator value_index_base() const
        {
            return this->value;
        };
        inline explicit constexpr operator rocsparseio_index_base() const
        {
            return (rocsparseio_index_base)this->value;
        };
        constexpr index_base_t(){};
        constexpr index_base_t(rocsparseio_index_base_t ival_)
            : value((value_index_base)ival_){};

        template <typename T>
        static constexpr index_base_t convert();
        inline bool                   is_invalid() const
        {
            switch(this->value)
            {
            case one:
            case zero:
            {
                return false;
            }
            }
            return true;
        };

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(one);
                CASE(zero);
#undef CASE
            }
            return "unknown";
        }
    };

    //!
    //! @brief c++11 struct for rwmode enum.
    //!
    struct rwmode_t
    {
        typedef enum value_type_ : rocsparseio_rwmode_t
        {
            read  = rocsparseio_rwmode_read,
            write = rocsparseio_rwmode_write
        } value_type;

        value_type value;

        inline constexpr operator value_type() const
        {
            return this->value;
        };

        inline constexpr rwmode_t(rocsparseio_rwmode_t ival_)
            : value((value_type)ival_)
        {
        }

        inline bool is_invalid() const
        {
            switch(this->value)
            {
            case read:
            case write:
            {
                return false;
            }
            }
            return true;
        };

        inline const char* to_string() const
        {
            switch(this->value)
            {
#define CASE(case_name)    \
    case case_name:        \
    {                      \
        return #case_name; \
    }
                CASE(read);
                CASE(write);
#undef CASE
            }
            return "unknown";
        }
    };

    inline std::ostream& operator<<(std::ostream& os_, const status_t& that_)
    {
        os_ << that_.to_string();
        return os_;
    }

    inline std::ostream& operator<<(std::ostream& os_, const direction_t& that_)
    {
        os_ << that_.to_string();
        return os_;
    }

    inline std::ostream& operator<<(std::ostream& os_, const order_t& that_)
    {
        os_ << that_.to_string();
        return os_;
    }

    inline std::ostream& operator<<(std::ostream& os_, const type_t& that_)
    {
        os_ << that_.to_string();
        return os_;
    }

    inline std::ostream& operator<<(std::ostream& os_, const rwmode_t& that_)
    {
        os_ << that_.to_string();
        return os_;
    }

    inline std::ostream& operator<<(std::ostream& os_, const format_t& that_)
    {
        os_ << that_.to_string();
        return os_;
    }

} // namespace rocsparseio

struct _rocsparseio_handle
{
    rocsparseio::rwmode_t mode;
    std::string           filename{};
    FILE*                 f{};
    _rocsparseio_handle(rocsparseio::rwmode_t mode_, const char* filename_)
        : mode(mode_)
        , filename(filename_)
    {
    }
};

namespace rocsparseio
{
    using handle_t = rocsparseio_handle;

    inline status_t fread_data(FILE* in_, uint64_t size_, uint64_t nmemb_, void* data_)
    {
        if(nmemb_ != fread(data_, size_, nmemb_, in_))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    template <typename source, typename target>
    inline status_t convert_scalar(const source& s_, target& t_)
    {
        if(!std::is_same<source, target>())
        {
            if(std::is_unsigned<source>())
            {
                if(sizeof(target) < sizeof(source))
                {
                    if(static_cast<source>(std::numeric_limits<target>::max()) < s_)
                    {
                        std::cerr << "convert out of bounds " << s_ << ", max is "
                                  << std::numeric_limits<target>::max() << "" << std::endl;
                        return status_t::invalid_value;
                    }
                }
            }
            else
            {
                if(sizeof(target) < sizeof(source))
                {
                    if((s_ > static_cast<source>(std::numeric_limits<target>::max()))
                       || (s_ < static_cast<source>(std::numeric_limits<target>::lowest())))
                    {
                        std::cerr << "convert out of bounds " << s_ << ", ["
                                  << std::numeric_limits<target>::lowest() << ", "
                                  << std::numeric_limits<target>::max() << "]" << std::endl;
                        return status_t::invalid_value;
                    }
                }
            }
        }
        t_ = static_cast<target>(s_);
        return status_t::success;
    }

    template <typename source>
    inline status_t convert_scalar(const source& s_, format_t& t_)
    {
        t_ = static_cast<format_t>(s_);
        if(t_.is_invalid())
        {
            return status_t::invalid_value;
        }
        return status_t::success;
    }

    template <typename source>
    inline status_t convert_scalar(const source& s_, direction_t& t_)
    {
        t_ = static_cast<direction_t>(s_);
        if(t_.is_invalid())
        {
            return status_t::invalid_value;
        }
        return status_t::success;
    }

    template <typename source>
    inline status_t convert_scalar(const source& s_, order_t& t_)
    {
        t_ = static_cast<order_t>(s_);
        if(t_.is_invalid())
        {
            return status_t::invalid_value;
        }
        return status_t::success;
    }

    template <typename source>
    inline status_t convert_scalar(const source& s_, type_t& t_)
    {
        t_ = static_cast<type_t>(s_);
        if(t_.is_invalid())
        {
            return status_t::invalid_value;
        }
        return status_t::success;
    }

    template <typename T, typename J>
    inline status_t fwrite_scalar(J scalar_, FILE* out_)
    {
        {
            T value = static_cast<T>(scalar_);
            if(1 != fwrite(&value, sizeof(T), 1, out_))
            {
                return status_t::invalid_file_operation;
            }
        }
        return status_t::success;
    };

    template <typename T, typename J>
    inline status_t fread_scalar(J& scalar_, FILE* in_)
    {
        {
            T value;
            if(1 != fread(&value, sizeof(T), 1, in_))
            {
                return status_t::invalid_file_operation;
            }
            return convert_scalar(value, scalar_);
        }
        return status_t::success;
    };

    template <typename T, typename J>
    inline status_t fread_scalar(J* scalar_, FILE* in_)
    {
        {
            T value;
            if(1 != fread(&value, sizeof(T), 1, in_))
            {
                return status_t::invalid_file_operation;
            }
            J        scalar;
            status_t status = convert_scalar(value, scalar);
            if(status != status_t::success)
            {
                return status;
            }
            scalar_[0] = scalar;
        }
        return status_t::success;
    };

    inline status_t fwrite_array(FILE* out_, uint64_t size_, uint64_t nmemb_, const void* data_)
    {
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(size_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(nmemb_, out_));
        if(nmemb_ != fwrite(data_, size_, nmemb_, out_))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    inline status_t fread_array(FILE* in_, void* data_)
    {
        uint64_t size;
        uint64_t nmemb;
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(size, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(nmemb, in_));
        if(nmemb != fread(data_, size, nmemb, in_))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    inline status_t fread_array_metadata(FILE* in_, uint64_t* size_, uint64_t* nmemb_)
    {
        const long pos = ftell(in_);
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(size_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(nmemb_, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

} // namespace rocsparseio

namespace rocsparseio
{

    //
    // ////////////////////////////////////////////////
    //
    inline status_t fwrite_dense_vector(FILE*       out_,
                                        type_t      data_type_,
                                        uint64_t    data_nmemb_,
                                        const void* data_,
                                        uint64_t    data_inc_,
                                        const char* name_,
                                        ...)
    {
        ROCSPARSEIO_CHECK_ARG(out_ == nullptr, status_t::invalid_pointer);
        ROCSPARSEIO_CHECK_ARG(data_type_.is_invalid(), status_t::invalid_value);
        ROCSPARSEIO_CHECK_ARG(((data_nmemb_ > 0) && (data_ == nullptr)), status_t::invalid_pointer);
        ROCSPARSEIO_CHECK_ARG(((data_inc_ > 0) && (data_ == nullptr)), status_t::invalid_size);
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(format_t::dense_vector, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(data_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(data_nmemb_, out_));

        rocsparseio_string name;
        if(name_)
        {
            va_list args;
            va_start(args, name_);
            if(vsnprintf(name, sizeof(rocsparseio_string), name_, args)
               >= sizeof(rocsparseio_string))
            {
                std::cerr << "the name of the object to save is too long and would be truncated. "
                          << std::endl;
                return status_t::invalid_value;
            }
            va_end(args);
        }
        else
        {
            sprintf(name, "unknown");
        }

        if(size_t(1) != fwrite(name, sizeof(rocsparseio_string), 1, out_))
        {
            return status_t::invalid_file_operation;
        }

        uint64_t data_size = data_type_.size();
        if(data_inc_ == 1)
        {
            if(data_nmemb_ != fwrite(data_, data_size, data_nmemb_, out_))
            {
                return status_t::invalid_file_operation;
            }
        }
        else
        {
            for(uint64_t i = 0; i < data_nmemb_; ++i)
            {
                if(1
                   != fwrite(((const char*)data_) + i * data_inc_ * data_size, data_size, 1, out_))
                {
                    return status_t::invalid_file_operation;
                }
            }
        }
        return status_t::success;
    }

    inline status_t fread_format(FILE* in_, format_t* format_)
    {
        ROCSPARSEIO_CHECK_ARG(in_ == nullptr, status_t::invalid_pointer);
        ROCSPARSEIO_CHECK_ARG(format_ == nullptr, status_t::invalid_pointer);
        const long pos = ftell(in_);

        if(0 != fseek(in_, sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(format_, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    };

    inline status_t fread_name(FILE* in_, rocsparseio_string name_)
    {
        ROCSPARSEIO_CHECK_ARG(in_ == nullptr, status_t::invalid_pointer);
        ROCSPARSEIO_CHECK_ARG(name_ == nullptr, status_t::invalid_pointer);
        const long pos = ftell(in_);

        if(size_t(1) != fread(name_, sizeof(rocsparseio_string), 1, in_))
        {
            return status_t::invalid_file_operation;
        }

        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    };

    inline status_t fread_metadata_dense_vector(FILE* in_, type_t* type_, uint64_t* nmemb_)
    {
        ROCSPARSEIO_CHECK_ARG(in_ == nullptr, status_t::invalid_pointer);
        ROCSPARSEIO_CHECK_ARG(type_ == nullptr, status_t::invalid_pointer);
        ROCSPARSEIO_CHECK_ARG(nmemb_ == nullptr, status_t::invalid_pointer);
        const long pos = ftell(in_);

        if(0 != fseek(in_, sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(format, in_));
        if(format != format_t::dense_vector)
        {
            std::cerr << " wrong format, not flagged as a dense_vector. " << std::endl;
            return status_t::invalid_format;
        }

        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(nmemb_, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    };

    inline status_t fread_dense_vector(FILE* in_, void* data_, uint64_t inc_)
    {
        if(0 != fseek(in_, sizeof(uint64_t) + sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }
        type_t type;
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(type, in_));
        uint64_t size = type.size();
        uint64_t nmemb;
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(nmemb, in_));
        if(inc_ == 1)
        {
            return fread_data(in_, size, nmemb, data_);
        }
        else
        {
            void* buff = malloc(size * nmemb);
            ROCSPARSEIO_CHECK(fread_data(in_, size, nmemb, buff));
            uint64_t j = 0;
            for(uint64_t i = 0; i < nmemb; ++i)
            {
                memcpy(((char*)data_) + j * size, ((char*)buff) + i * size, size);
                j += inc_;
            }
            free(buff);
            return status_t::invalid_value;
        }
    }

} // namespace rocsparseio

namespace rocsparseio
{

    inline status_t fwrite_dense_matrix(FILE*       out_,
                                        order_t     order_,
                                        uint64_t    m_,
                                        uint64_t    n_,
                                        type_t      data_type_,
                                        const void* data_,
                                        uint64_t    data_ld_,
                                        const char* name_,
                                        ...)
    {
        ROCSPARSEIO_CHECK_ARG(out_ == nullptr, status_t::invalid_pointer);

        ROCSPARSEIO_CHECK_ARG(order_.is_invalid(), status_t::invalid_value);

        ROCSPARSEIO_CHECK_ARG(data_type_.is_invalid(), status_t::invalid_value);

        ROCSPARSEIO_CHECK_ARG((((m_ > 0) && (n_ > 0)) && (data_ == nullptr)),
                              status_t::invalid_pointer);

        ROCSPARSEIO_CHECK_ARG((((m_ > 0) && (n_ > 0)) && (data_ == nullptr)),
                              status_t::invalid_pointer);

        ROCSPARSEIO_CHECK_ARG(((order_ == order_t::row) && (data_ld_ < n_)),
                              status_t::invalid_size);

        ROCSPARSEIO_CHECK_ARG(((order_ == order_t::column) && (data_ld_ < m_)),
                              status_t::invalid_size);

        rocsparseio_string name;
        if(name_)
        {
            va_list args;
            va_start(args, name_);
            if(vsnprintf(name, sizeof(rocsparseio_string), name_, args)
               >= sizeof(rocsparseio_string))
            {
                std::cerr << "the name of the object to save is too long and would be truncated. "
                          << std::endl;
                return status_t::invalid_value;
            }
            va_end(args);
        }
        else
        {
            sprintf(name, "unknown");
        }

        if(size_t(1) != fwrite(name, sizeof(rocsparseio_string), 1, out_))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(format_t::dense_matrix, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(order_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(m_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(n_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(data_type_, out_));

        uint64_t data_size = data_type_.size();
        switch(order_)
        {
        case order_t::row:
        {
            if(data_ld_ == n_)
            {
                const uint64_t data_nmemb = m_ * n_;
                if(data_nmemb != fwrite(data_, data_size, data_nmemb, out_))
                {
                    return status_t::invalid_file_operation;
                }
            }
            else
            {
                for(uint64_t i = 0; i < m_; ++i)
                {
                    if(n_
                       != fwrite(
                           ((const char*)data_) + i * data_ld_ * data_size, data_size, n_, out_))
                    {
                        return status_t::invalid_file_operation;
                    }
                }
            }
            break;
        }

        case order_t::column:
        {
            if(data_ld_ == m_)
            {
                const uint64_t data_nmemb = m_ * n_;
                if(data_nmemb != fwrite(data_, data_size, data_nmemb, out_))
                {
                    return status_t::invalid_file_operation;
                }
            }
            else
            {
                for(uint64_t j = 0; j < n_; ++j)
                {
                    if(m_
                       != fwrite(
                           ((const char*)data_) + j * data_ld_ * data_size, data_size, m_, out_))
                    {
                        return status_t::invalid_file_operation;
                    }
                }
            }
            break;
        }
        }
        return status_t::success;
    };

    inline status_t fread_dense_matrix(FILE* in_, void* data_, uint64_t ld_)
    {
        if(0 != fseek(in_, sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        {
            format_t format(0);
            ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(format, in_));
            if(format != format_t::dense_matrix)
            {
                std::cerr << " wrong format, not flagged as a dense matrix. " << std::endl;
                return status_t::invalid_format;
            }
        }

        order_t order;
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(&order, in_));
        uint64_t m;
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(&m, in_));
        uint64_t n;
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(&n, in_));
        type_t type;
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(&type, in_));
        const uint64_t s = type.size();

        ROCSPARSEIO_CHECK_ARG(((order == order_t::row) && (ld_ < n)), status_t::invalid_size);
        ROCSPARSEIO_CHECK_ARG(((order == order_t::column) && (ld_ < m)), status_t::invalid_size);

        switch(order)
        {
        case order_t::row:
        {
            if(ld_ == n)
            {
                ROCSPARSEIO_CHECK(fread_data(in_, s, m * n, data_));
            }
            else
            {
                char* p = static_cast<char*>(data_);
                for(uint64_t i = 0; i < m; ++i)
                {
                    ROCSPARSEIO_CHECK(fread_data(in_, s, n, p));
                    p += s * ld_;
                }
            }
            break;
        }
        case order_t::column:
        {
            if(ld_ == m)
            {
                ROCSPARSEIO_CHECK(fread_data(in_, s, m * n, data_));
            }
            else
            {
                char*          p = static_cast<char*>(data_);
                const uint64_t s = type.size();
                for(uint64_t j = 0; j < n; ++j)
                {
                    ROCSPARSEIO_CHECK(fread_data(in_, s, m, p));
                    p += s * ld_;
                }
            }
            break;
        }
        }

        return status_t::success;
    }

    inline status_t fread_metadata_dense_matrix(
        FILE* in_, order_t* order_, uint64_t* m_, uint64_t* n_, type_t* type_)
    {
        const long pos = ftell(in_);

        if(0 != fseek(in_, sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(format, in_));
        if(format != format_t::dense_matrix)
        {
            std::cerr << " wrong format, not flagged as a dense matrix. " << std::endl;
            return status_t::invalid_format;
        }

        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(order_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(m_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(n_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(type_, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

} // namespace rocsparseio

//
//
//
namespace rocsparseio
{
    inline status_t fwrite_sparse_csx(FILE*       out_,
                                      direction_t dir_,
                                      uint64_t    m_,
                                      uint64_t    n_,
                                      uint64_t    nnz_,
                                      type_t      ptr_type_,
                                      const void* __restrict__ ptr_,
                                      type_t ind_type_,
                                      const void* __restrict__ ind_,
                                      type_t data_type_,
                                      const void* __restrict__ data_,
                                      index_base_t base_,
                                      const char*  name_,
                                      ...)
    {

        rocsparseio_string name;
        if(name_)
        {
            va_list args;
            va_start(args, name_);
            if(vsnprintf(name, sizeof(rocsparseio_string), name_, args)
               >= sizeof(rocsparseio_string))
            {
                std::cerr << "the name of the object to save is too long and would be truncated. "
                          << std::endl;
                return status_t::invalid_value;
            }
            va_end(args);
        }
        else
        {
            sprintf(name, "unknown");
        }

        if(size_t(1) != fwrite(name, sizeof(rocsparseio_string), 1, out_))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(format_t::sparse_csx, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(dir_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(m_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(n_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(nnz_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ptr_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(data_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(base_, out_));
        switch(dir_)
        {
        case direction_t::row:
        {
            ROCSPARSEIO_CHECK(fwrite_array(out_, ptr_type_.size(), m_ + 1, ptr_));
            break;
        }
        case direction_t::column:
        {
            ROCSPARSEIO_CHECK(fwrite_array(out_, ptr_type_.size(), n_ + 1, ptr_));
            break;
        }
        }

        ROCSPARSEIO_CHECK(fwrite_array(out_, ind_type_.size(), nnz_, ind_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, data_type_.size(), nnz_, data_));

        return status_t::success;
    };

    inline status_t fread_metadata_sparse_csx(FILE*         in_,
                                              direction_t*  dir_,
                                              uint64_t*     m_,
                                              uint64_t*     n_,
                                              uint64_t*     nnz_,
                                              type_t*       ptr_type_,
                                              type_t*       ind_type_,
                                              type_t*       data_type_,
                                              index_base_t* base_)
    {
        const long pos = ftell(in_);

        if(0 != fseek(in_, sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(format, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(dir_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(m_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(n_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(nnz_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ptr_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ind_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(data_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(base_, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }

        return status_t::success;
    }

    inline status_t fread_sparse_csx(FILE* in_,
                                     void* __restrict__ ptr_,
                                     void* __restrict__ ind_,
                                     void* __restrict__ data_)
    {
        if(0 != fseek(in_, sizeof(uint64_t) * 9 + sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fread_array(in_, ptr_));
        ROCSPARSEIO_CHECK(fread_array(in_, ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, data_));

        return status_t::success;
    }

} // namespace rocsparseio

namespace rocsparseio
{
    inline status_t fwrite_sparse_coo(FILE*    out_,
                                      uint64_t m_,
                                      uint64_t n_,
                                      uint64_t nnz_,
                                      type_t   row_ind_type_,
                                      const void* __restrict__ row_ind_,
                                      type_t col_ind_type_,
                                      const void* __restrict__ col_ind_,
                                      type_t data_type_,
                                      const void* __restrict__ data_,
                                      index_base_t base_,
                                      const char*  name_,
                                      ...)
    {

        rocsparseio_string name;
        if(name_)
        {
            va_list args;
            va_start(args, name_);
            if(vsnprintf(name, sizeof(rocsparseio_string), name_, args)
               >= sizeof(rocsparseio_string))
            {
                std::cerr << "the name of the object to save is too long and would be truncated. "
                          << std::endl;
                return status_t::invalid_value;
            }
            va_end(args);
        }
        else
        {
            sprintf(name, "unknown");
        }

        if(size_t(1) != fwrite(name, sizeof(rocsparseio_string), 1, out_))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(format_t::sparse_coo, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(m_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(n_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(nnz_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(row_ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(col_ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(data_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(base_, out_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, row_ind_type_.size(), nnz_, row_ind_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, col_ind_type_.size(), nnz_, col_ind_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, data_type_.size(), nnz_, data_));
        return status_t::success;
    };

    inline status_t fread_metadata_sparse_coo(FILE*         in_,
                                              uint64_t*     m_,
                                              uint64_t*     n_,
                                              uint64_t*     nnz_,
                                              type_t*       row_ind_type_,
                                              type_t*       col_ind_type_,
                                              type_t*       data_type_,
                                              index_base_t* base_)
    {
        const long pos = ftell(in_);

        if(0 != fseek(in_, sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(format, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(m_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(n_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(nnz_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(row_ind_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(col_ind_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(data_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(base_, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    inline status_t fread_sparse_coo(FILE* in_,
                                     void* __restrict__ row_ind_,
                                     void* __restrict__ col_ind_,
                                     void* __restrict__ data_)
    {
        if(0 != fseek(in_, sizeof(uint64_t) * 8 + sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fread_array(in_, row_ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, col_ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, data_));
        return status_t::success;
    }

} // namespace rocsparseio

namespace rocsparseio
{
    inline status_t fwrite_sparse_ell(FILE*    out_,
                                      uint64_t m_,
                                      uint64_t n_,
                                      uint64_t width_,
                                      type_t   ind_type_,
                                      const void* __restrict__ ind_,
                                      type_t val_type_,
                                      const void* __restrict__ val_,
                                      index_base_t base_,
                                      const char*  name_,
                                      ...)
    {
        rocsparseio_string name;
        if(name_)
        {
            va_list args;
            va_start(args, name_);
            if(vsnprintf(name, sizeof(rocsparseio_string), name_, args)
               >= sizeof(rocsparseio_string))
            {
                std::cerr << "the name of the object to save is too long and would be truncated. "
                          << std::endl;
                return status_t::invalid_value;
            }
            va_end(args);
        }
        else
        {
            sprintf(name, "unknown");
        }

        if(size_t(1) != fwrite(name, sizeof(rocsparseio_string), 1, out_))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(format_t::sparse_ell, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(m_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(n_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(width_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(val_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(base_, out_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, ind_type_.size(), m_ * width_, ind_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, val_type_.size(), m_ * width_, val_));
        return status_t::success;
    };

    inline status_t fread_metadata_sparse_ell(FILE*         in_,
                                              uint64_t*     m_,
                                              uint64_t*     n_,
                                              uint64_t*     width_,
                                              type_t*       ind_type_,
                                              type_t*       val_type_,
                                              index_base_t* base_)
    {
        const long pos = ftell(in_);

        if(0 != fseek(in_, sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(format, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(m_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(n_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(width_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ind_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(val_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(base_, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    inline status_t fread_sparse_ell(FILE* in_, void* __restrict__ ind_, void* __restrict__ val_)
    {
        if(0 != fseek(in_, sizeof(uint64_t) * 7 + sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fread_array(in_, ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, val_));

        return status_t::success;
    }

} // namespace rocsparseio

namespace rocsparseio
{
    inline status_t fwrite_sparse_dia(FILE*    out_,
                                      uint64_t m_,
                                      uint64_t n_,
                                      uint64_t ndiag_,
                                      type_t   ind_type_,
                                      const void* __restrict__ ind_,
                                      type_t val_type_,
                                      const void* __restrict__ val_,
                                      index_base_t base_,
                                      const char*  name_,
                                      ...)
    {
        rocsparseio_string name;
        if(name_)
        {
            va_list args;
            va_start(args, name_);
            if(vsnprintf(name, sizeof(rocsparseio_string), name_, args)
               >= sizeof(rocsparseio_string))
            {
                std::cerr << "the name of the object to save is too long and would be truncated. "
                          << std::endl;
                return status_t::invalid_value;
            }
            va_end(args);
        }
        else
        {
            sprintf(name, "unknown");
        }

        if(size_t(1) != fwrite(name, sizeof(rocsparseio_string), 1, out_))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(format_t::sparse_dia, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(m_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(n_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ndiag_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(val_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(base_, out_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, ind_type_.size(), ndiag_, ind_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, val_type_.size(), std::min(m_, n_) * ndiag_, val_));
        return status_t::success;
    };

    inline status_t fread_metadata_sparse_dia(FILE*         in_,
                                              uint64_t*     m_,
                                              uint64_t*     n_,
                                              uint64_t*     ndiag_,
                                              type_t*       ind_type_,
                                              type_t*       val_type_,
                                              index_base_t* base_)
    {
        const long pos = ftell(in_);

        if(0 != fseek(in_, sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(format, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(m_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(n_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ndiag_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ind_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(val_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(base_, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    inline status_t fread_sparse_dia(FILE* in_, void* __restrict__ ind_, void* __restrict__ val_)
    {
        if(0 != fseek(in_, sizeof(uint64_t) * 7 + sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fread_array(in_, ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, val_));

        return status_t::success;
    }

} // namespace rocsparseio

namespace rocsparseio
{

    inline status_t fwrite_sparse_hyb(FILE*    out_,
                                      uint64_t m_,
                                      uint64_t n_,

                                      uint64_t coo_nnz_,
                                      type_t   coo_row_ind_type_,
                                      const void* __restrict__ coo_row_ind_,
                                      type_t coo_col_ind_type_,
                                      const void* __restrict__ coo_col_ind_,
                                      type_t coo_data_type_,
                                      const void* __restrict__ coo_data_,
                                      index_base_t coo_base_,

                                      uint64_t ell_width_,
                                      type_t   ell_ind_type_,
                                      const void* __restrict__ ell_ind_,
                                      type_t ell_val_type_,
                                      const void* __restrict__ ell_val_,
                                      index_base_t ell_base_,
                                      const char*  name_,
                                      ...)
    {

        rocsparseio_string name;
        if(name_)
        {
            va_list args;
            va_start(args, name_);
            if(vsnprintf(name, sizeof(rocsparseio_string), name_, args)
               >= sizeof(rocsparseio_string))
            {
                std::cerr << "the name of the object to save is too long and would be truncated. "
                          << std::endl;
                return status_t::invalid_value;
            }
            va_end(args);
        }
        else
        {
            sprintf(name, "unknown");
        }

        if(size_t(1) != fwrite(name, sizeof(rocsparseio_string), 1, out_))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(format_t::sparse_hyb, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(m_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(n_, out_));

        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(coo_nnz_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(coo_row_ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(coo_col_ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(coo_data_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(coo_base_, out_));

        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ell_width_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ell_ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ell_val_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ell_base_, out_));

        ROCSPARSEIO_CHECK(fwrite_array(out_, coo_row_ind_type_.size(), coo_nnz_, coo_row_ind_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, coo_col_ind_type_.size(), coo_nnz_, coo_col_ind_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, coo_data_type_.size(), coo_nnz_, coo_data_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, ell_ind_type_.size(), m_ * ell_width_, ell_ind_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, ell_val_type_.size(), m_ * ell_width_, ell_val_));

        return status_t::success;
    };

    inline status_t fread_metadata_sparse_hyb(FILE*         in_,
                                              uint64_t*     m_,
                                              uint64_t*     n_,
                                              uint64_t*     coo_nnz_,
                                              type_t*       coo_row_ind_type_,
                                              type_t*       coo_col_ind_type_,
                                              type_t*       coo_data_type_,
                                              index_base_t* coo_base_,
                                              uint64_t*     ell_width_,
                                              type_t*       ell_ind_type_,
                                              type_t*       ell_val_type_,
                                              index_base_t* ell_base_)
    {
        const long pos = ftell(in_);

        if(0 != fseek(in_, sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(format, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(m_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(n_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(coo_nnz_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(coo_row_ind_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(coo_col_ind_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(coo_data_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(coo_base_, in_));

        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ell_width_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ell_ind_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ell_val_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ell_base_, in_));

        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    inline status_t fread_sparse_hyb(FILE* in_,
                                     void* __restrict__ coo_row_ind_,
                                     void* __restrict__ coo_col_ind_,
                                     void* __restrict__ coo_data_,
                                     void* __restrict__ ell_ind_,
                                     void* __restrict__ ell_val_)
    {
        if(0 != fseek(in_, sizeof(uint64_t) * 12 + sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fread_array(in_, coo_row_ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, coo_col_ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, coo_data_));

        ROCSPARSEIO_CHECK(fread_array(in_, ell_ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, ell_val_));
        return status_t::success;
    }

} // namespace rocsparseio

namespace rocsparseio
{

    inline status_t fwrite_sparse_gebsx(FILE*       out_,
                                        direction_t dir_,
                                        direction_t dirb_,
                                        uint64_t    mb_,
                                        uint64_t    nb_,
                                        uint64_t    nnzb_,
                                        uint64_t    row_block_dim_,
                                        uint64_t    col_block_dim_,
                                        type_t      ptr_type_,
                                        const void* __restrict__ ptr_,
                                        type_t ind_type_,
                                        const void* __restrict__ ind_,
                                        type_t data_type_,
                                        const void* __restrict__ data_,
                                        index_base_t base_,
                                        const char*  name_,
                                        ...)
    {

        rocsparseio_string name;
        if(name_)
        {
            va_list args;
            va_start(args, name_);
            if(vsnprintf(name, sizeof(rocsparseio_string), name_, args)
               >= sizeof(rocsparseio_string))
            {
                std::cerr << "the name of the object to save is too long and would be truncated. "
                          << std::endl;
                return status_t::invalid_value;
            }
            va_end(args);
        }
        else
        {
            sprintf(name, "unknown");
        }

        if(size_t(1) != fwrite(name, sizeof(rocsparseio_string), 1, out_))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(format_t::sparse_gebsx, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(dir_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(dirb_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(mb_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(nb_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(nnzb_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(row_block_dim_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(col_block_dim_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ptr_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(data_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(base_, out_));

        switch(dir_)
        {
        case direction_t::row:
        {
            ROCSPARSEIO_CHECK(fwrite_array(out_, ptr_type_.size(), mb_ + 1, ptr_));
            break;
        }
        case direction_t::column:
        {
            ROCSPARSEIO_CHECK(fwrite_array(out_, ptr_type_.size(), nb_ + 1, ptr_));
            break;
        }
        }

        ROCSPARSEIO_CHECK(fwrite_array(out_, ind_type_.size(), nnzb_, ind_));
        ROCSPARSEIO_CHECK(
            fwrite_array(out_, data_type_.size(), nnzb_ * row_block_dim_ * col_block_dim_, data_));

        return status_t::success;
    };

    inline status_t fread_metadata_sparse_gebsx(FILE*         in_,
                                                direction_t*  dir_,
                                                direction_t*  dirb_,
                                                uint64_t*     mb_,
                                                uint64_t*     nb_,
                                                uint64_t*     nnzb_,
                                                uint64_t*     row_block_dim_,
                                                uint64_t*     col_block_dim_,
                                                type_t*       ptr_type_,
                                                type_t*       ind_type_,
                                                type_t*       data_type_,
                                                index_base_t* base_)
    {
        const long pos = ftell(in_);

        if(0 != fseek(in_, sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(format, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(dir_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(dirb_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(mb_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(nb_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(nnzb_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(row_block_dim_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(col_block_dim_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ptr_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ind_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(data_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(base_, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }
        return status_t::success;
    }

    inline status_t fread_sparse_gebsx(FILE* in_,
                                       void* __restrict__ ptr_,
                                       void* __restrict__ ind_,
                                       void* __restrict__ data_)
    {
        if(0 != fseek(in_, sizeof(uint64_t) * 12 + sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fread_array(in_, ptr_));
        ROCSPARSEIO_CHECK(fread_array(in_, ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, data_));

        return status_t::success;
    }

} // namespace rocsparseio

//
//
//
namespace rocsparseio
{
    inline status_t fwrite_sparse_mcsx(FILE*       out_,
                                       direction_t dir_,
                                       uint64_t    m_,
                                       uint64_t    n_,
                                       uint64_t    nnz_,
                                       type_t      ptr_type_,
                                       const void* __restrict__ ptr_,
                                       type_t ind_type_,
                                       const void* __restrict__ ind_,
                                       type_t data_type_,
                                       const void* __restrict__ data_,
                                       index_base_t base_,
                                       const char*  name_,
                                       ...)
    {

        rocsparseio_string name;
        if(name_)
        {
            va_list args;
            va_start(args, name_);
            if(vsnprintf(name, sizeof(rocsparseio_string), name_, args)
               >= sizeof(rocsparseio_string))
            {
                std::cerr << "the name of the object to save is too long and would be truncated. "
                          << std::endl;
                return status_t::invalid_value;
            }
            va_end(args);
        }
        else
        {
            sprintf(name, "unknown");
        }

        if(size_t(1) != fwrite(name, sizeof(rocsparseio_string), 1, out_))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(format_t::sparse_mcsx, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(dir_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(m_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(n_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(nnz_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ptr_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(ind_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(data_type_, out_));
        ROCSPARSEIO_CHECK(fwrite_scalar<uint64_t>(base_, out_));
        switch(dir_)
        {
        case direction_t::row:
        {
            ROCSPARSEIO_CHECK(fwrite_array(out_, ptr_type_.size(), m_ + 1, ptr_));
            break;
        }
        case direction_t::column:
        {
            ROCSPARSEIO_CHECK(fwrite_array(out_, ptr_type_.size(), n_ + 1, ptr_));
            break;
        }
        }

        ROCSPARSEIO_CHECK(fwrite_array(out_, ind_type_.size(), nnz_, ind_));
        ROCSPARSEIO_CHECK(fwrite_array(out_, data_type_.size(), nnz_, data_));

        return status_t::success;
    };

    inline status_t fread_metadata_sparse_mcsx(FILE*         in_,
                                               direction_t*  dir_,
                                               uint64_t*     m_,
                                               uint64_t*     n_,
                                               uint64_t*     nnz_,
                                               type_t*       ptr_type_,
                                               type_t*       ind_type_,
                                               type_t*       data_type_,
                                               index_base_t* base_)
    {
        const long pos = ftell(in_);

        if(0 != fseek(in_, sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        format_t format(0);
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(format, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(dir_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(m_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(n_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(nnz_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ptr_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(ind_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(data_type_, in_));
        ROCSPARSEIO_CHECK(fread_scalar<uint64_t>(base_, in_));
        if(0 != fseek(in_, pos, SEEK_SET))
        {
            return status_t::invalid_file_operation;
        }

        return status_t::success;
    }

    inline status_t fread_sparse_mcsx(FILE* in_,
                                      void* __restrict__ ptr_,
                                      void* __restrict__ ind_,
                                      void* __restrict__ data_)
    {
        if(0 != fseek(in_, sizeof(uint64_t) * 9 + sizeof(rocsparseio_string), SEEK_CUR))
        {
            return status_t::invalid_file_operation;
        }

        ROCSPARSEIO_CHECK(fread_array(in_, ptr_));
        ROCSPARSEIO_CHECK(fread_array(in_, ind_));
        ROCSPARSEIO_CHECK(fread_array(in_, data_));

        return status_t::success;
    }

} // namespace rocsparseio

namespace rocsparseio
{
    template <>
    inline type_t type_t::convert<int32_t>()
    {
        return type_t::int32;
    };
    template <>
    inline type_t type_t::convert<int64_t>()
    {
        return type_t::int64;
    };

    template <>
    inline type_t type_t::convert<float>()
    {
        return type_t::float32;
    };
    template <>
    inline type_t type_t::convert<double>()
    {
        return type_t::float64;
    };

    template <typename... Ts>
    inline status_t read_format(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_format(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_name(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_name(handle_->f, ts_...);
    }

    inline status_t open(rocsparseio_handle* p_handle_, rwmode_t mode, const char* filename, ...)
    {
        char filename_[512];
        {
            va_list args;
            va_start(args, filename);
            if(vsnprintf(filename_, 512, filename, args) >= 512)
            {
                std::cerr << "string is too long and is truncated. " << std::endl;
                return status_t::invalid_value;
            }
            va_end(args);
        }

        rocsparseio_handle h = new _rocsparseio_handle(mode, filename_);
        p_handle_[0]         = h;

        h->f = nullptr;
        switch(mode)
        {
        case rwmode_t::read:
        {
            h->f = fopen(filename_, "rb");
            if(h->f == nullptr)
            {
                return status_t::invalid_file;
            }

            {
                uint64_t ref_value[2]{};
                char*    p = (char*)&ref_value[0];
                sprintf(p, "ROCSPARSEIO.%d", ROCSPARSEIO_VERSION_MAJOR);
                uint64_t value[2]{};
                if(2 != fread(&value[0], sizeof(uint64_t), 2, h->f))
                {
                    return status_t::invalid_file_operation;
                }
                //	    std::cout << "read " << value[0] << " " << value[1] <<
                //std::endl;
                if(ref_value[0] != value[0] || ref_value[1] != value[1])
                {
                    std::cerr << "incompatible rocsparseio version: " << std::endl;
                    std::cerr << "   expected      : " << ref_value[0] << "." << ref_value[1]
                              << std::endl;
                    std::cerr << "   from file     : " << value[0] << "." << value[1] << std::endl;
                    return status_t::invalid_file;
                }
            }

            break;
        }

        case rwmode_t::write:
        {

            h->f = fopen(filename_, "wb");
            if(h->f == nullptr)
            {
                return status_t::invalid_file;
            }

            //
            // WRITE HEADER
            //
            {
                uint64_t value[2]{};
                char*    p = (char*)&value;
                sprintf(p, "ROCSPARSEIO.%d", ROCSPARSEIO_VERSION_MAJOR);
                if(2 != fwrite(&value[0], sizeof(uint64_t), 2, h->f))
                {
                    return status_t::invalid_file_operation;
                }
            }

            break;
        }
        }
        return status_t::success;
    }

    //!
    //! @brief Close.
    //!
    inline status_t close(rocsparseio_handle handle_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        if(handle_->f != nullptr)
        {
            fclose(handle_->f);
        }
        delete handle_;
        return status_t::success;
    }

    template <typename... Ts>
    inline status_t write_dense_vector(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fwrite_dense_vector(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_dense_vector(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_dense_vector(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_metadata_dense_vector(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_metadata_dense_vector(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t write_dense_matrix(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fwrite_dense_matrix(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_metadata_dense_matrix(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_metadata_dense_matrix(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_dense_matrix(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_dense_matrix(handle_->f, ts_...);
    }

    template <typename T, typename J>
    inline status_t write_dense_matrix_template(
        rocsparseio_handle handle_, order_t order_, J m_, J n_, const T* data_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        ROCSPARSEIO_CHECK_ARG(handle_->mode != rwmode_t::write, status_t::invalid_mode);
        return fwrite_dense_matrix(handle_->f,
                                   order_,
                                   static_cast<uint64_t>(m_),
                                   static_cast<uint64_t>(n_),
                                   type_t::convert<T>(),
                                   data_,
                                   (order_ == ROCSPARSEIO_ORDER_ROW) ? static_cast<uint64_t>(n_)
                                                                     : static_cast<uint64_t>(m_));
    };

    template <typename... Ts>
    inline status_t write_sparse_csx(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fwrite_sparse_csx(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_metadata_sparse_csx(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_metadata_sparse_csx(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_sparse_csx(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_sparse_csx(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t write_sparse_coo(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fwrite_sparse_coo(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_metadata_sparse_coo(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_metadata_sparse_coo(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_sparse_coo(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_sparse_coo(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t write_sparse_hyb(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fwrite_sparse_hyb(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_metadata_sparse_hyb(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_metadata_sparse_hyb(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_sparse_hyb(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_sparse_hyb(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t write_sparse_gebsx(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fwrite_sparse_gebsx(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_metadata_sparse_gebsx(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_metadata_sparse_gebsx(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_sparse_gebsx(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_sparse_gebsx(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t write_sparse_ell(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fwrite_sparse_ell(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_metadata_sparse_ell(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_metadata_sparse_ell(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_sparse_ell(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_sparse_ell(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t write_sparse_dia(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fwrite_sparse_dia(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_metadata_sparse_dia(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_metadata_sparse_dia(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_sparse_dia(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_sparse_dia(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t write_sparse_mcsx(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fwrite_sparse_mcsx(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_metadata_sparse_mcsx(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_metadata_sparse_mcsx(handle_->f, ts_...);
    }

    template <typename... Ts>
    inline status_t read_sparse_mcsx(rocsparseio_handle handle_, Ts&&... ts_)
    {
        ROCSPARSEIO_CHECK_ARG(!handle_, status_t::invalid_handle);
        return fread_sparse_mcsx(handle_->f, ts_...);
    }

} // namespace rocsparseio
