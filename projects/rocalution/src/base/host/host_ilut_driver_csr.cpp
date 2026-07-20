/* ************************************************************************
 * Copyright (C) 2023 Advanced Micro Devices, Inc. All rights Reserved.
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

#include "host_ilut_driver_csr.hpp"
#include "../../utils/def.hpp"

#include <algorithm>
#include <complex>
#include <cstring>

namespace rocalution
{

    template <typename T, typename J>
    ILUTDriverCSR<T, J>::ILUTDriverCSR(J m, J bw)
        : m(m)
        , bw(bw)
    {
        this->w_size = std::min(m, bw);
    }
    template <typename T, typename J>
    size_t ILUTDriverCSR<T, J>::buffer_size()
    {
        size_t b_size = 0;
        b_size += ((sizeof(J) * this->m - 1) / 256 + 1) * 256;
        b_size += ((sizeof(J) * this->w_size) / 256 + 1) * 256;

        // pad for alignment
        b_size += ((sizeof(T) * this->w_size) / 256 + 1) * 256;

        return b_size;
    }

    template <typename T, typename J>
    void ILUTDriverCSR<T, J>::set_buffer(void* buffer)
    {
        /*
        *   Distribute buffer
        */
        char* ptr = reinterpret_cast<char*>(buffer);

        this->jr = reinterpret_cast<J*>(ptr);
        ptr += ((sizeof(J) * this->m - 1) / 256 + 1) * 256;

        this->jw = reinterpret_cast<J*>(ptr);
        ptr += ((sizeof(J) * this->w_size) / 256 + 1) * 256;

        this->w = reinterpret_cast<T*>(ptr);

        std::memset(this->jr, 0, sizeof(J) * this->m);
    }

    template <typename T, typename J>
    void ILUTDriverCSR<T, J>::add_element(J col, T val)
    {
        // save lower
        if(col < this->row)
        {
            assert(this->len_l < this->w_size);

            this->jw[this->len_l] = col;
            this->w[this->len_l]  = val;
            this->jr[col]         = ++this->len_l;
        }
        // save diag
        else if(col == this->row)
        {
            this->jw[this->diag_pos] = col;
            this->w[this->diag_pos]  = val;
            this->jr[col]            = this->diag_pos + 1;
        }
        // save upper
        else
        {
            const J at = (this->diag_pos + 1) + this->len_u;
            assert(at < this->w_size);

            this->jw[at] = col;
            this->w[at]  = val;

            this->jr[col] = at + 1;

            this->len_u++;
        }
    }

    template <typename T, typename J>
    void ILUTDriverCSR<T, J>::swap(J index1, J index2)
    {
        // swap jw
        {
            J temp           = this->jw[index1];
            this->jw[index1] = this->jw[index2];
            this->jw[index2] = temp;
        }

        // swap w
        {
            T temp          = this->w[index1];
            this->w[index1] = this->w[index2];
            this->w[index2] = temp;
        }
    }

    template <typename T, typename J>
    void ILUTDriverCSR<T, J>::initialize(const T* val, const J* col, J size, int base, J row)
    {
        this->row           = row;
        this->len_l         = 0;
        this->len_u         = 0;
        this->stored_size_l = 0;
        this->stored_size_u = 0;
        this->front_work_l  = 0;
        this->norm          = 0;

        /*
        *   Copy Elements
        */
        this->diag_pos = (this->bw == this->m) ? this->row : this->bw / 2;

        for(J i = 0; i < size; i++)
        {
            // J k = col[i];
            const T v = val[i];
            const J c = col[i] - base;

            // save lower
            if(c < this->row)
            {
                assert(this->len_l < this->w_size);

                this->jw[this->len_l] = c;
                this->w[this->len_l]  = v;
                this->jr[c]           = ++this->len_l;
            }
            // save diag
            else if(c == this->row)
            {
                this->jw[this->diag_pos] = c;
                this->w[this->diag_pos]  = v;
                this->jr[c]              = this->diag_pos + 1;
            }
            // save upper
            else
            {
                const J at = (this->diag_pos + 1) + this->len_u;
                assert(at < this->w_size);

                this->jw[at] = c;
                this->w[at]  = v;

                this->jr[c] = at + 1;

                this->len_u++;
            }
            this->norm += std::abs(v);
        }
        this->norm /= size;
    }

    template <typename T, typename J>
    bool ILUTDriverCSR<T, J>::next_lower(J& col, T& val)
    {
        // quick return if the next column is not a lower element
        if(this->front_work_l == this->len_l)
        {
            return false;
        }

        // find smallest column in remaining working lower elements
        J min_col_index = this->front_work_l;
        for(J i = min_col_index + 1; i < this->len_l; i++)
        {
            if(this->jw[i] < this->jw[min_col_index])
            {
                min_col_index = i;
            }
        }

        // swap lowest column to front
        if(min_col_index != this->front_work_l)
        {
            swap(this->front_work_l, min_col_index);
            J temp                                 = this->jr[this->jw[this->front_work_l]];
            this->jr[this->jw[this->front_work_l]] = this->jr[this->jw[min_col_index]];
            this->jr[this->jw[min_col_index]]      = temp;
        }

        col = this->jw[this->front_work_l];
        val = this->w[this->front_work_l];

        this->jr[col] = 0;

        this->front_work_l++;

        return true;
    }

    template <typename T, typename J>
    void ILUTDriverCSR<T, J>::save_lower(T val)
    {
        J to_save = this->front_work_l - 1;

        // swap
        this->jw[this->stored_size_l] = this->jw[to_save];
        this->w[this->stored_size_l]  = val;

        this->stored_size_l++;
    }

    template <typename T, typename J>
    void ILUTDriverCSR<T, J>::add_to_element(J col, T val_to_add)
    {
        J pos = this->jr[col];

        // if not fill in
        if(pos != 0)
        {
            this->w[pos - 1] += val_to_add;
        }
        // if fill in
        else
        {
            add_element(col, val_to_add);
        }
    }

    template <typename T, typename J>
    void ILUTDriverCSR<T, J>::partition(J begin, J size, J p)
    {
        if(size > p && p > 0)
        {
            J left  = begin;
            J right = left + size;
            while(left < right)
            {
                J pivot   = left;
                J part_in = left + 1;

                for(J i = left + 1; i < right; i++)
                {
                    if(std::abs(this->w[i]) > std::abs(this->w[pivot]))
                    {
                        swap(i, part_in);

                        part_in++;
                    }
                }
                {
                    // part_in points to beginning of right partition
                    // move pivot to end of the left partition
                    swap(pivot, part_in - 1);

                    pivot = part_in - 1;
                }
                if((pivot - begin) == p)
                {
                    break;
                }
                else if((pivot - begin) > p)
                {
                    // right is exclusive
                    right = pivot;
                }
                else
                {
                    // left is inclusive
                    left = pivot + 1;
                }
            }
        }
    }

    template <typename T, typename J>
    void ILUTDriverCSR<T, J>::trim(double tol, J p)
    {
        // filter/fix diagonal element
        if(this->jr[this->row] == 0 || this->w[this->diag_pos] == static_cast<T>(0))
        {
            this->jr[this->row]      = this->diag_pos + 1;
            this->w[this->diag_pos]  = (0.0001f + std::abs(tol)) * this->norm;
            this->jw[this->diag_pos] = this->row;
        }

        // filter upper
        double r_tol        = std::abs(tol) * this->norm;
        this->stored_size_u = 0;
        for(J i = 0; i < this->len_u; i++)
        {
            J front_index = this->diag_pos + 1 + this->stored_size_u;
            J curr_index  = this->diag_pos + 1 + i;

            this->jr[this->jw[curr_index]] = 0;

            // if passes criteria swap to front
            if(std::abs(this->w[curr_index]) > r_tol)
            {
                swap(front_index, curr_index);
                this->stored_size_u++;
            }
        }

        // partition
        partition(0, this->stored_size_l, p);
        this->stored_size_l = std::min(this->stored_size_l, p);

        partition(this->diag_pos + 1, this->stored_size_u, p - 1);
        this->stored_size_u = std::min(this->stored_size_u, p - 1);
    }

    template <typename T, typename J>
    J ILUTDriverCSR<T, J>::row_size()
    {
        J size = this->stored_size_l + this->stored_size_u;
        size += (this->jr[this->row] == 0) ? 0 : 1;

        return size;
    }

    template <typename T, typename J>
    bool ILUTDriverCSR<T, J>::store_row(T* val, J* col, J& diag_in)
    {
        bool status = true;

        J i;
        for(i = 0; i < this->stored_size_l; i++)
        {
            val[i] = this->w[i];
            col[i] = this->jw[i];
        }

        if(this->jr[this->row] != 0)
        {
            diag_in = this->stored_size_l;

            val[i] = this->w[this->diag_pos];
            col[i] = this->jw[this->diag_pos];
            i++;

            this->jr[this->row] = 0;
        }
        else
        {
            status = false;
        }

        for(int j = 0; j < this->stored_size_u; j++)
        {
            J index = this->diag_pos + 1 + j;

            val[i] = this->w[index];
            col[i] = this->jw[index];
            i++;
        }

        return status;
    }

    template class ILUTDriverCSR<float, int>;
    template class ILUTDriverCSR<double, int>;
#ifdef SUPPORT_COMPLEX
    template class ILUTDriverCSR<std::complex<float>, int>;
    template class ILUTDriverCSR<std::complex<double>, int>;
#endif

} // namespace rocalution
