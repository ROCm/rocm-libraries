// Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef FFT_PARAMS_H
#define FFT_PARAMS_H

#include <algorithm>
#include <hip/hip_runtime.h>
#include <iostream>
#include <mutex>
#include <numeric>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <random>
#include <tuple>
#include <unordered_set>
#include <valarray>
#include <vector>

#include "../shared/arithmetic.h"
#include "../shared/array_validator.h"
#include "../shared/client_data_layout_helpers.h"
#include "../shared/data_gen_device.h"
#include "../shared/data_gen_host.h"
#include "../shared/device_properties.h"
#include "../shared/fft_enums.h"
#include "../shared/gpubuf.h"
#include "../shared/printbuffer.h"
#include "../shared/ptrdiff.h"
#include "../shared/rocfft_complex.h"

// Used for CLI11 parsing of precision enum
static bool lexical_cast(const std::string& word, fft_precision& precision)
{
    if(word == "half")
        precision = fft_precision_half;
    else if(word == "single")
        precision = fft_precision_single;
    else if(word == "double")
        precision = fft_precision_double;
    else
        throw std::runtime_error("Invalid precision specified");
    return true;
}

// Used for CLI11 parsing of auto-allocation enum
static bool lexical_cast(const std::string& word, fft_auto_allocation& auto_allocation)
{
    if(word == "on")
        auto_allocation = fft_auto_allocation_on;
    else if(word == "off")
        auto_allocation = fft_auto_allocation_off;
    else if(word == "default")
        auto_allocation = fft_auto_allocation_default;
    else
        throw std::runtime_error(
            "Invalid auto-allocation behavior specified (choose \"on\", \"off\", or \"default\")");
    return true;
}

// Used for CLI11 parsing of input gen enum
static bool lexical_cast(const std::string& word, fft_input_generator& gen)
{
    if(word == "0")
        gen = fft_input_random_generator_device;
    else if(word == "1")
        gen = fft_input_random_generator_host;
    else if(word == "2")
        gen = fft_input_generator_device;
    else if(word == "3")
        gen = fft_input_generator_host;
    else
        throw std::runtime_error("Invalid input generator specified");
#ifndef USE_HIPRAND
    if(gen == fft_input_random_generator_device)
        throw std::runtime_error(
            "Device random input generation is not available, as hipRAND support is not enabled");
#endif
    return true;
}

// Determine the size of the data type given the precision and type.
template <typename Tsize>
inline Tsize var_size(const fft_precision precision, const fft_array_type type)
{
    size_t var_size = 0;
    switch(precision)
    {
    case fft_precision_half:
        var_size = sizeof(rocfft_fp16);
        break;
    case fft_precision_single:
        var_size = sizeof(float);
        break;
    case fft_precision_double:
        var_size = sizeof(double);
        break;
    }
    switch(type)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
        var_size *= 2;
        break;
    default:
        break;
    }
    return var_size;
}

// Given an array type and transform length, strides, etc, initialize
// values into the input device buffer.
//
// length/istride/batch/dist describe the physical layout of the
// input buffer.  The buffer is treated as a sub-brick of a field,
// though the brick may cover the entire field.
//
// Lower coordinate of the brick in the field is provided by
// field_lower (FFT dimension coordinate) and field_lower_batch
// (batch dimension coordinate).  For a brick that covers the whole
// field, these are all zeroes.
//
// field_contig_stride + dist are the field's stride and dist if the
// field were contiguous.
template <typename Tfloat, typename Tint1>
inline void set_input(std::vector<gpubuf>&       input,
                      const fft_input_generator  igen,
                      const fft_array_type       itype,
                      const std::vector<size_t>& length,
                      const std::vector<size_t>& ilength,
                      const std::vector<size_t>& istride,
                      const std::vector<size_t>& ioffset,
                      const Tint1&               whole_length,
                      const Tint1&               whole_stride,
                      const size_t               idist,
                      const size_t               nbatch,
                      const hipDeviceProp_t&     deviceProp,
                      const Tint1&               field_lower,
                      const size_t               field_lower_batch,
                      const Tint1&               field_contig_stride,
                      const size_t               field_contig_dist)
{
#ifndef USE_HIPRAND
    if(igen == fft_input_random_generator_device)
        throw std::runtime_error(
            "Device random input generation is not available, as hipRAND support is not enabled");
#endif // USE_HIPRAND

    if(is_host_generator(igen))
        throw std::runtime_error("Host input generation is not available for gpu buffers");

    auto isize = count_iters(whole_length) * nbatch;

    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        auto ibuffer = (rocfft_complex<Tfloat>*)input[0].data() + ioffset[0];
#ifdef USE_HIPRAND
        if(igen == fft_input_generator_device)
            generate_interleaved_data(
                whole_length, idist, isize, whole_stride, nbatch, ibuffer, deviceProp);
        else if(igen == fft_input_random_generator_device)
            generate_random_interleaved_data(whole_length,
                                             idist,
                                             isize,
                                             whole_stride,
                                             ibuffer,
                                             deviceProp,
                                             field_lower,
                                             field_lower_batch,
                                             field_contig_stride,
                                             field_contig_dist);
#else
        generate_interleaved_data(
            whole_length, idist, isize, whole_stride, nbatch, ibuffer, deviceProp);
#endif // USE_HIPRAND

        if(itype == fft_array_type_hermitian_interleaved)
        {
            auto ibuffer_2 = (rocfft_complex<Tfloat>*)input[0].data();
            impose_hermitian_symmetry_interleaved(
                length, ilength, istride, idist, nbatch, ibuffer_2, deviceProp);
        }

        break;
    }
    case fft_array_type_complex_planar:
    case fft_array_type_hermitian_planar:
    {
        auto ibuffer_real = (Tfloat*)input[0].data() + ioffset[0];
        auto ibuffer_imag = (Tfloat*)input[1].data() + ioffset[1];

#ifdef USE_HIPRAND
        if(igen == fft_input_generator_device)
            generate_planar_data(whole_length,
                                 idist,
                                 isize,
                                 whole_stride,
                                 nbatch,
                                 ibuffer_real,
                                 ibuffer_imag,
                                 deviceProp);
        else if(igen == fft_input_random_generator_device)
            generate_random_planar_data(whole_length,
                                        idist,
                                        isize,
                                        whole_stride,
                                        ibuffer_real,
                                        ibuffer_imag,
                                        deviceProp,
                                        field_lower,
                                        field_lower_batch,
                                        field_contig_stride,
                                        field_contig_dist);
#else
        generate_planar_data(whole_length,
                             idist,
                             isize,
                             whole_stride,
                             nbatch,
                             ibuffer_real,
                             ibuffer_imag,
                             deviceProp);
#endif // USE_HIPRAND

        if(itype == fft_array_type_hermitian_planar)
            impose_hermitian_symmetry_planar(
                length, ilength, istride, idist, nbatch, ibuffer_real, ibuffer_imag, deviceProp);

        break;
    }
    case fft_array_type_real:
    {
        auto ibuffer = (Tfloat*)input[0].data() + ioffset[0];

#ifdef USE_HIPRAND
        if(igen == fft_input_generator_device)
            generate_real_data(
                whole_length, idist, isize, whole_stride, nbatch, ibuffer, deviceProp);
        else if(igen == fft_input_random_generator_device)
            generate_random_real_data(whole_length,
                                      idist,
                                      isize,
                                      whole_stride,
                                      ibuffer,
                                      deviceProp,
                                      field_lower,
                                      field_lower_batch,
                                      field_contig_stride,
                                      field_contig_dist);
#else
        generate_real_data(whole_length, idist, isize, whole_stride, nbatch, ibuffer, deviceProp);
#endif // USE_HIPRAND

        break;
    }
    default:
        throw std::runtime_error("Input layout format not yet supported");
    }
}

// Given an array type and transform length, strides, etc, initialize
// values into the input host buffer.
//
// length/istride/batch/dist describe the physical layout of the
// input buffer.  The buffer is treated as a sub-brick of a field,
// though the brick may cover the entire field.
//
// Lower coordinate of the brick in the field is provided by
// field_lower (FFT dimension coordinate) and field_lower_batch
// (batch dimension coordinate).  For a brick that covers the whole
// field, these are all zeroes.
//
// field_contig_stride + dist are the field's stride and dist if the
// field were contiguous.
template <typename Tfloat, typename Tint1>
inline void set_input(std::vector<hostbuf>&      input,
                      const fft_input_generator  igen,
                      const fft_array_type       itype,
                      const std::vector<size_t>& length,
                      const std::vector<size_t>& ilength,
                      const std::vector<size_t>& istride,
                      const std::vector<size_t>& ioffset,
                      const Tint1&               whole_length,
                      const Tint1&               whole_stride,
                      const size_t               idist,
                      const size_t               nbatch,
                      const hipDeviceProp_t&     deviceProp,
                      const Tint1                field_lower,
                      const size_t               field_lower_batch,
                      const Tint1                field_contig_stride,
                      const size_t               field_contig_dist)
{
    if(is_device_generator(igen))
        throw std::runtime_error(
            "Device random input generation is not available for host buffers");

    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        if(igen == fft_input_generator_host)
            generate_interleaved_data<Tfloat>(
                input, ioffset, whole_length, whole_stride, idist, nbatch);
        else if(igen == fft_input_random_generator_host)
            generate_random_interleaved_data<Tfloat>(input,
                                                     ioffset,
                                                     whole_length,
                                                     whole_stride,
                                                     idist,
                                                     nbatch,
                                                     field_lower,
                                                     field_lower_batch,
                                                     field_contig_stride,
                                                     field_contig_dist);

        if(itype == fft_array_type_hermitian_interleaved)
            impose_hermitian_symmetry_interleaved<Tfloat>(
                input, ioffset, length, istride, idist, nbatch);

        break;
    }
    case fft_array_type_complex_planar:
    case fft_array_type_hermitian_planar:
    {
        if(igen == fft_input_generator_host)
            generate_planar_data<Tfloat>(input, ioffset, whole_length, whole_stride, idist, nbatch);
        else if(igen == fft_input_random_generator_host)
            generate_random_planar_data<Tfloat>(input,
                                                ioffset,
                                                whole_length,
                                                whole_stride,
                                                idist,
                                                nbatch,
                                                field_lower,
                                                field_lower_batch,
                                                field_contig_stride,
                                                field_contig_dist);

        if(itype == fft_array_type_hermitian_planar)
            impose_hermitian_symmetry_planar<Tfloat>(
                input, ioffset, length, istride, idist, nbatch);

        break;
    }
    case fft_array_type_real:
    {
        if(igen == fft_input_generator_host)
            generate_real_data<Tfloat>(input, ioffset, whole_length, whole_stride, idist, nbatch);
        else if(igen == fft_input_random_generator_host)
            generate_random_real_data<Tfloat>(input,
                                              ioffset,
                                              whole_length,
                                              whole_stride,
                                              idist,
                                              nbatch,
                                              field_lower,
                                              field_lower_batch,
                                              field_contig_stride,
                                              field_contig_dist);

        break;
    }
    default:
        throw std::runtime_error("Input layout format not yet supported");
    }
}

// unroll set_input for dimension 1, 2, 3
template <typename Tbuff, typename Tfloat>
inline void set_input(std::vector<Tbuff>&        input,
                      const fft_input_generator  igen,
                      const fft_array_type       itype,
                      const std::vector<size_t>& length,
                      const std::vector<size_t>& ilength,
                      const std::vector<size_t>& istride,
                      const std::vector<size_t>& ioffset,
                      const size_t               idist,
                      const size_t               nbatch,
                      const hipDeviceProp_t&     deviceProp,
                      const std::vector<size_t>& field_lower,
                      const size_t               field_lower_batch,
                      const std::vector<size_t>& field_contig_stride,
                      const size_t               field_contig_dist)
{
    switch(length.size())
    {
    case 1:
        set_input<Tfloat, size_t>(input,
                                  igen,
                                  itype,
                                  length,
                                  ilength,
                                  istride,
                                  ioffset,
                                  ilength[0],
                                  istride[0],
                                  idist,
                                  nbatch,
                                  deviceProp,
                                  field_lower[0],
                                  field_lower_batch,
                                  field_contig_stride[0],
                                  field_contig_dist);
        break;
    case 2:
        set_input<Tfloat, std::tuple<size_t, size_t>>(
            input,
            igen,
            itype,
            length,
            ilength,
            istride,
            ioffset,
            std::make_tuple(ilength[0], ilength[1]),
            std::make_tuple(istride[0], istride[1]),
            idist,
            nbatch,
            deviceProp,
            std::make_tuple(field_lower[0], field_lower[1]),
            field_lower_batch,
            std::make_tuple(field_contig_stride[0], field_contig_stride[1]),
            field_contig_dist);
        break;
    case 3:
        set_input<Tfloat, std::tuple<size_t, size_t, size_t>>(
            input,
            igen,
            itype,
            length,
            ilength,
            istride,
            ioffset,
            std::make_tuple(ilength[0], ilength[1], ilength[2]),
            std::make_tuple(istride[0], istride[1], istride[2]),
            idist,
            nbatch,
            deviceProp,
            std::make_tuple(field_lower[0], field_lower[1], field_lower[2]),
            field_lower_batch,
            std::make_tuple(field_contig_stride[0], field_contig_stride[1], field_contig_stride[2]),
            field_contig_dist);
        break;
    default:
        abort();
    }
}

// Container class for test parameters.
class fft_params
{
public:
    // All parameters are row-major.
    std::vector<size_t>  length;
    std::vector<size_t>  istride;
    std::vector<size_t>  ostride;
    size_t               nbatch         = 1;
    fft_precision        precision      = fft_precision_single;
    fft_transform_type   transform_type = fft_transform_type_complex_forward;
    fft_result_placement placement      = fft_placement_inplace;
    size_t               idist          = 0;
    size_t               odist          = 0;
    fft_array_type       itype          = fft_array_type_unset;
    fft_array_type       otype          = fft_array_type_unset;
    std::vector<size_t>  ioffset        = {0, 0};
    std::vector<size_t>  ooffset        = {0, 0};

    std::vector<size_t> isize;
    std::vector<size_t> osize;

#ifdef USE_HIPRAND
    fft_input_generator igen = fft_input_random_generator_device;
#else
    fft_input_generator igen = fft_input_random_generator_host;
#endif

    fft_auto_allocation auto_allocate = fft_auto_allocation_default;

    enum fft_mp_lib
    {
        fft_mp_lib_none,
        fft_mp_lib_mpi,
    };
    fft_mp_lib mp_lib = fft_mp_lib_none;
    // Pointer to a library-specific communicator type.  Note that this
    // is a pointer, so whatever this points to must live as long as
    // this pointer does.
    void* mp_comm = nullptr;

    struct fft_brick
    {
        // all vectors here are row-major, with same length as FFT
        // dimension + 1 (for batch dimension)

        // inclusive lower bound of brick
        std::vector<size_t> lower;
        // exclusive upper bound of brick
        std::vector<size_t> upper;
        // stride of brick in memory
        std::vector<size_t> stride;

        // compute the length of this brick
        std::vector<size_t> length() const
        {
            std::vector<size_t> ret;
            for(size_t i = 0; i < lower.size(); ++i)
                ret.push_back(upper[i] - lower[i]);
            return ret;
        }

        // compute offset of lower bound in a field with the given
        // stride + dist (batch stride is separate)
        size_t lower_field_offset(std::vector<size_t> stride, size_t dist) const
        {
            // brick strides include batch, so adjust our input accordingly
            stride.insert(stride.begin(), dist);

            return std::inner_product(lower.begin(), lower.end(), stride.begin(), 0);
        }

        // location of the brick
        int rank   = 0;
        int device = 0;

        bool operator==(const fft_brick& other) const
        {
            return this->lower == other.lower && this->upper == other.upper
                   && this->stride == other.stride && this->rank == other.rank
                   && this->device == other.device;
        }
        bool operator!=(const fft_brick& other) const
        {
            return !(*this == other);
        }
    };

    struct fft_field
    {
        std::vector<fft_brick> bricks;

        // Brick ordering is significant - a rank needs to pass
        // pointers to execute() in the order that the bricks were
        // added to the field.
        //
        // But we also want to sort the bricks by rank, so that we can
        // efficiently find the bricks for a given rank.  So do a
        // stable sort to preserve the relative order of bricks on any
        // rank, even after they're sorted.
        void stable_sort_by_rank()
        {
            std::stable_sort(
                bricks.begin(), bricks.end(), [](const fft_brick& a, const fft_brick& b) {
                    return a.rank < b.rank;
                });
        }
    };

    // heuristic algorithm to create a 3D grid that covers a field starting in index (0,0,0),
    // this approach intends to minimize the surface area of field bricks
    void set_default_3d_grid(unsigned int               mp_ranks,
                             std::vector<size_t> const  grid_dims,
                             std::vector<unsigned int>& fft_grid)
    {
        std::valarray<unsigned int> global_indices = {static_cast<unsigned int>(grid_dims[0]),
                                                      static_cast<unsigned int>(grid_dims[1]),
                                                      static_cast<unsigned int>(grid_dims[2])};

        // set initial grid as ones
        std::valarray<unsigned int> selected_grid = {1, 1, 1};

        // helper method to compute the surface of a brick
        auto surface = [&](std::valarray<unsigned int> const& proc_grid) -> unsigned int {
            auto brick_size = global_indices / proc_grid;
            return (brick_size * brick_size.cshift(1)).sum();
        };

        unsigned int selected_surface = std::numeric_limits<unsigned int>::max();

        for(unsigned int i = 1; i <= mp_ranks; i++)
        {
            if(mp_ranks % i == 0)
            {
                unsigned int remainder = mp_ranks / i;
                for(unsigned int j = 1; j <= remainder; j++)
                {
                    if(remainder % j == 0)
                    {
                        std::valarray<unsigned int> grid = {i, j, remainder / j};
                        unsigned int const          surf = surface(grid);
                        if(surf < selected_surface)
                        {
                            selected_surface = surf;
                            selected_grid    = grid;
                        }
                    }
                }
            }
        }

        if(selected_grid[0] * selected_grid[1] * selected_grid[2] != mp_ranks)
        {
            throw std::runtime_error("Grid dimensions do not multiply to mp_ranks.");
        }

        fft_grid = {selected_grid[0], selected_grid[1], selected_grid[2]};
    }

    // heuristic algorithm to create a 2D grid that covers a field starting in index (0,0),
    // this approach intends to minimize the surface area of field bricks
    void set_default_2d_grid(unsigned int               mp_ranks,
                             std::vector<size_t> const  grid_dims,
                             std::vector<unsigned int>& fft_grid)
    {
        std::valarray<unsigned int> global_indices
            = {static_cast<unsigned int>(grid_dims[0]), static_cast<unsigned int>(grid_dims[1])};

        // set initial grid as ones
        std::valarray<unsigned int> selected_grid = {1, 1};

        // helper method to compute the surface of a brick
        auto surface = [&](std::valarray<unsigned int> const& proc_grid) -> unsigned int {
            auto brick_size = global_indices / proc_grid;
            return (brick_size * brick_size.cshift(1)).sum();
        };

        unsigned int selected_surface = std::numeric_limits<unsigned int>::max();

        for(unsigned int i = 1; i <= mp_ranks; i++)
        {
            if(mp_ranks % i == 0)
            {
                std::valarray<unsigned int> grid = {i, mp_ranks / i};
                unsigned int const          surf = surface(grid);
                if(surf < selected_surface)
                {
                    selected_surface = surf;
                    selected_grid    = grid;
                }
            }
        }

        if(selected_grid[0] * selected_grid[1] != mp_ranks)
        {
            throw std::runtime_error("Grid dimensions do not multiply to mp_ranks.");
        }

        fft_grid = {selected_grid[0], selected_grid[1]};
    }

    void set_default_grid(const int&                 mp_ranks,
                          std::vector<unsigned int>& ingrid,
                          std::vector<unsigned int>& outgrid)
    {
        if(ingrid.empty())
        {
            if(length.size() == 3)
            {
                set_default_3d_grid(mp_ranks, length, ingrid);
            }
            else if(length.size() == 2)
            {
                set_default_2d_grid(mp_ranks, length, ingrid);
            }
            else if(length.size() == 1)
            {
                ingrid.push_back(mp_ranks);
            }
        }
        if(outgrid.empty())
        {
            if(length.size() == 3)
            {
                set_default_3d_grid(mp_ranks, length, outgrid);
            }
            else if(length.size() == 2)
            {
                set_default_2d_grid(mp_ranks, length, outgrid);
            }
            else if(length.size() == 1)
            {
                outgrid.push_back(mp_ranks);
            }
        }

        // sanity checks
        int ingrid_size  = product(ingrid.begin(), ingrid.end());
        int outgrid_size = product(outgrid.begin(), outgrid.end());

        if((ingrid.size() != length.size()) || (outgrid.size() != length.size()))
        {
            throw std::runtime_error(
                "Grid of processors must be of the same dimension as the FFT!");
        }

        if((ingrid_size != mp_ranks) || (outgrid_size != mp_ranks))
        {
            throw std::runtime_error("Number of GPUs defined by input/output grids must be "
                                     "equal to the number of available GPUs!");
        }
    }

    // optional brick decomposition of inputs/outputs
    std::vector<fft_field> ifields;
    std::vector<fft_field> ofields;

    // Check that a supplied vector of callback pointers has an
    // expected size.  Optionally also check that each pointer is
    // non-null.  Throws an exception if a check fails.  The vector
    // itself can be null, as callbacks are optional.
    static void check_callback_vec(std::vector<void*>* cb, size_t expected_size, bool nonnull)
    {
        if(!cb)
            return;
        if(cb->size() != expected_size)
            throw std::invalid_argument("expected " + std::to_string(expected_size)
                                        + " callback pointers, got " + std::to_string(cb->size()));
        if(nonnull && std::any_of(cb->begin(), cb->end(), [](void* p) { return p == nullptr; }))
            throw std::invalid_argument("null callback function");
    }

    // simple "multi-GPU" count, meaning the library decides on the
    // decomposition instead of it being explicit as bricks.  only
    // has an effect if set to a number > 1.
    size_t multiGPU = 0;

    // run testing load/store callbacks
    bool                    run_callbacks   = false;
    static constexpr double load_cb_scalar  = 0.457813941;
    static constexpr double store_cb_scalar = 0.391504938;

    // Check that data outside of output strides is not overwritten.
    // This is only set explicitly on some tests where there's space
    // between dimensions, but the dimensions are still in-order.
    // We're not trying to generically find holes in arbitrary data
    // layouts.
    //
    // NOTE: this flag is not included in tokens, since it doesn't
    // affect how the FFT library behaves.
    bool check_output_strides = false;

    // scaling factor - we do a pointwise multiplication of outputs by
    // this factor
    double scale_factor = 1.0;

    fft_params(){};
    virtual ~fft_params(){};

    // copying and moving
    fft_params(const fft_params&) = default;
    fft_params& operator=(const fft_params&) = default;
    fft_params(fft_params&&)                 = default;
    fft_params& operator=(fft_params&&) = default;

    virtual void setup() {}
    virtual void cleanup() {}

    // Given an array type, return the name as a string.
    static std::string array_type_name(const fft_array_type type, bool verbose = true)
    {
        switch(type)
        {
        case fft_array_type_complex_interleaved:
            return verbose ? "fft_array_type_complex_interleaved" : "CI";
        case fft_array_type_complex_planar:
            return verbose ? "fft_array_type_complex_planar" : "CP";
        case fft_array_type_real:
            return verbose ? "fft_array_type_real" : "R";
        case fft_array_type_hermitian_interleaved:
            return verbose ? "fft_array_type_hermitian_interleaved" : "HI";
        case fft_array_type_hermitian_planar:
            return verbose ? "fft_array_type_hermitian_planar" : "HP";
        case fft_array_type_unset:
            return verbose ? "fft_array_type_unset" : "UN";
        }
        return "";
    }

    std::string transform_type_name() const
    {
        switch(transform_type)
        {
        case fft_transform_type_complex_forward:
            return "fft_transform_type_complex_forward";
        case fft_transform_type_complex_inverse:
            return "fft_transform_type_complex_inverse";
        case fft_transform_type_real_forward:
            return "fft_transform_type_real_forward";
        case fft_transform_type_real_inverse:
            return "fft_transform_type_real_inverse";
        default:
            throw std::runtime_error("Invalid transform type");
        }
    }

    bool is_inverse() const
    {
        return is_bwd(transform_type);
    }

    bool is_forward() const
    {
        return is_fwd(transform_type);
    }

    // Convert to string for output.
    std::string str(const std::string& separator = ", ") const
    {
        // top-level stride/dist are not used when fields are specified.
        const bool have_ifields = !ifields.empty();
        const bool have_ofields = !ofields.empty();

        std::stringstream ss;
        auto print_size_vec = [&](const char* description, const std::vector<size_t>& vec) {
            ss << description << ":";
            for(auto i : vec)
                ss << " " << i;
            ss << separator;
        };
        auto print_fields = [&](const char* description, const std::vector<fft_field>& fields) {
            for(unsigned int fidx = 0; fidx < fields.size(); ++fidx)
            {
                const auto& f = fields[fidx];
                ss << description << " " << fidx << ":" << separator;
                for(unsigned int bidx = 0; bidx < f.bricks.size(); ++bidx)
                {
                    const auto& b = f.bricks[bidx];
                    ss << " brick " << bidx << ":" << separator;
                    print_size_vec("  lower", b.lower);
                    print_size_vec("  upper", b.upper);
                    print_size_vec("  stride", b.stride);
                    ss << "  device: " << b.device << separator;
                }
            }
        };

        print_size_vec("length", length);
        if(have_ifields)
        {
            print_fields("ifield", ifields);
        }
        else
        {
            print_size_vec("istride", istride);
            ss << "idist: " << idist << separator;
        }

        if(have_ofields)
        {
            print_fields("ofield", ofields);
        }
        else
        {
            print_size_vec("ostride", ostride);
            ss << "odist: " << odist << separator;
        }

        ss << "batch: " << nbatch << separator;
        print_size_vec("isize", isize);
        print_size_vec("osize", osize);

        print_size_vec("ioffset", ioffset);
        print_size_vec("ooffset", ooffset);

        if(placement == fft_placement_inplace)
            ss << "in-place";
        else
            ss << "out-of-place";
        ss << separator;
        ss << "transform_type: " << transform_type_name() << separator;
        ss << array_type_name(itype) << " -> " << array_type_name(otype) << separator;
        switch(precision)
        {
        case fft_precision_half:
            ss << "half-precision";
            break;
        case fft_precision_single:
            ss << "single-precision";
            break;
        case fft_precision_double:
            ss << "double-precision";
            break;
        }
        ss << separator;

        print_size_vec("ilength", ilength());
        print_size_vec("olength", olength());

        print_size_vec("ibuffer_size", ibuffer_sizes());
        print_size_vec("obuffer_size", obuffer_sizes());

        if(scale_factor != 1.0)
            ss << "scale factor: " << scale_factor << separator;

        return ss.str();
    }

    // Produce a stringified token of the test fft params.
    std::string token() const
    {
        std::string ret;

        switch(transform_type)
        {
        case fft_transform_type_complex_forward:
            ret += "complex_forward_";
            break;
        case fft_transform_type_complex_inverse:
            ret += "complex_inverse_";
            break;
        case fft_transform_type_real_forward:
            ret += "real_forward_";
            break;
        case fft_transform_type_real_inverse:
            ret += "real_inverse_";
            break;
        }

        auto append_size_vec = [&ret](const std::vector<size_t>& vec) {
            for(auto s : vec)
            {
                ret += "_";
                ret += std::to_string(s);
            }
        };

        ret += "len";
        append_size_vec(length);

        switch(precision)
        {
        case fft_precision_half:
            ret += "_half_";
            break;
        case fft_precision_single:
            ret += "_single_";
            break;
        case fft_precision_double:
            ret += "_double_";
            break;
        }

        switch(placement)
        {
        case fft_placement_inplace:
            ret += "ip_";
            break;
        case fft_placement_notinplace:
            ret += "op_";
            break;
        }

        ret += "batch_";
        ret += std::to_string(nbatch);

        auto append_array_type = [&ret](fft_array_type type) {
            switch(type)
            {
            case fft_array_type_complex_interleaved:
                ret += "CI";
                break;
            case fft_array_type_complex_planar:
                ret += "CP";
                break;
            case fft_array_type_real:
                ret += "R";
                break;
            case fft_array_type_hermitian_interleaved:
                ret += "HI";
                break;
            case fft_array_type_hermitian_planar:
                ret += "HP";
                break;
            default:
                ret += "UN";
                break;
            }
        };

        auto append_brick_info = [&ret, &append_size_vec](const fft_brick& b) {
            ret += "_brick";

            ret += "_lower";
            append_size_vec(b.lower);
            ret += "_upper";
            append_size_vec(b.upper);
            ret += "_stride";
            append_size_vec(b.stride);
            if(b.rank)
            {
                ret += "_rank_";
                ret += std::to_string(b.rank);
            }
            ret += "_dev_";
            ret += std::to_string(b.device);
        };

        const bool have_ifields = !ifields.empty();
        const bool have_ofields = !ofields.empty();

        if(have_ifields)
        {
            for(const auto& f : ifields)
            {
                ret += "_ifield";
                for(const auto& b : f.bricks)
                    append_brick_info(b);
            }
        }
        else
        {
            ret += "_istride";
            append_size_vec(istride);
            ret += "_";
            append_array_type(itype);
        }

        if(have_ofields)
        {
            for(const auto& f : ofields)
            {
                ret += "_ofield";
                for(const auto& b : f.bricks)
                    append_brick_info(b);
            }
        }
        else
        {
            ret += "_ostride";
            append_size_vec(ostride);
            ret += "_";
            append_array_type(otype);
        }

        if(!have_ifields)
        {
            ret += "_idist_";
            ret += std::to_string(idist);
        }
        if(!have_ofields)
        {
            ret += "_odist_";
            ret += std::to_string(odist);
        }

        if(!have_ifields)
        {
            ret += "_ioffset";
            append_size_vec(ioffset);
        }

        if(!have_ofields)
        {
            ret += "_ooffset";
            append_size_vec(ooffset);
        }

        if(run_callbacks)
            ret += "_CB";

        if(scale_factor != 1.0)
            ret += "_scale";

        if(multiGPU > 1)
        {
            ret += "_multigpu_";
            ret += std::to_string(multiGPU);
        }

        if(auto_allocate != fft_auto_allocation_default)
        {
            ret += "_autoallocation_";
            ret += (auto_allocate == fft_auto_allocation_on ? "on" : "off");
        }

        return ret;
    }

    // Set all params from a stringified token.
    void from_token(std::string token)
    {
        std::vector<std::string> vals;

        std::string delimiter = "_";
        {
            size_t pos = 0;
            while((pos = token.find(delimiter)) != std::string::npos)
            {
                auto val = token.substr(0, pos);
                vals.push_back(val);
                token.erase(0, pos + delimiter.length());
            }
            vals.push_back(token);
        }

        auto size_parser
            = [](const std::vector<std::string>& vals, const std::string token, size_t& pos) {
                  if(vals[pos++] != token)
                      throw std::runtime_error("Unable to parse token");
                  return std::stoull(vals[pos++]);
              };

        auto vector_parser
            = [](const std::vector<std::string>& vals, const std::string token, size_t& pos) {
                  if(vals[pos++] != token)
                      throw std::runtime_error("Unable to parse token");
                  std::vector<size_t> vec;

                  while(pos < vals.size())
                  {
                      if(std::all_of(vals[pos].begin(), vals[pos].end(), ::isdigit))
                      {
                          vec.push_back(std::stoull(vals[pos++]));
                      }
                      else
                      {
                          break;
                      }
                  }
                  return vec;
              };

        auto type_parser = [](const std::string& val) {
            if(val == "CI")
                return fft_array_type_complex_interleaved;
            else if(val == "CP")
                return fft_array_type_complex_planar;
            else if(val == "R")
                return fft_array_type_real;
            else if(val == "HI")
                return fft_array_type_hermitian_interleaved;
            else if(val == "HP")
                return fft_array_type_hermitian_planar;
            return fft_array_type_unset;
        };

        auto field_parser = [&vector_parser, &size_parser](const std::vector<std::string>& vals,
                                                           size_t&                         pos,
                                                           std::vector<fft_field>&         output) {
            // skip over ifield/ofield word
            pos++;
            fft_field& f = output.emplace_back();
            while(pos < vals.size() && vals[pos] == "brick")
            {
                fft_brick& b = f.bricks.emplace_back();
                pos++;
                b.lower  = vector_parser(vals, "lower", pos);
                b.upper  = vector_parser(vals, "upper", pos);
                b.stride = vector_parser(vals, "stride", pos);
                if(vals[pos] == "rank")
                    b.rank = size_parser(vals, "rank", pos);
                b.device = size_parser(vals, "dev", pos);
            }
        };

        size_t pos = 0;

        bool complex = vals[pos++] == "complex";
        bool forward = vals[pos++] == "forward";

        if(complex && forward)
            transform_type = fft_transform_type_complex_forward;
        if(complex && !forward)
            transform_type = fft_transform_type_complex_inverse;
        if(!complex && forward)
            transform_type = fft_transform_type_real_forward;
        if(!complex && !forward)
            transform_type = fft_transform_type_real_inverse;

        length = vector_parser(vals, "len", pos);

        if(vals[pos] == "half")
            precision = fft_precision_half;
        else if(vals[pos] == "single")
            precision = fft_precision_single;
        else if(vals[pos] == "double")
            precision = fft_precision_double;
        pos++;

        placement = (vals[pos++] == "ip") ? fft_placement_inplace : fft_placement_notinplace;

        nbatch = size_parser(vals, "batch", pos);

        // strides, bricks etc are mixed in from here, so just keep
        // looking at the next token to decide what to do
        while(pos < vals.size() - 1)
        {
            const auto& next_token = vals[pos];
            if(next_token == "istride")
            {
                istride = vector_parser(vals, "istride", pos);
                itype   = type_parser(vals[pos]);
                pos++;
            }
            else if(next_token == "ostride")
            {
                ostride = vector_parser(vals, "ostride", pos);
                otype   = type_parser(vals[pos]);
                pos++;
            }
            else if(next_token == "idist")
                idist = size_parser(vals, "idist", pos);
            else if(next_token == "odist")
                odist = size_parser(vals, "odist", pos);
            else if(next_token == "ioffset")
                ioffset = vector_parser(vals, "ioffset", pos);
            else if(next_token == "ooffset")
                ooffset = vector_parser(vals, "ooffset", pos);
            else if(next_token == "ifield")
                field_parser(vals, pos, ifields);
            else if(next_token == "ofield")
                field_parser(vals, pos, ofields);
            else
                break;
        }

        if(pos < vals.size() && vals[pos] == "CB")
        {
            run_callbacks = true;
            ++pos;
        }

        if(pos < vals.size() && vals[pos] == "scale")
        {
            // just pick some factor that's not zero or one
            scale_factor = 0.1239;
            ++pos;
        }

        if(pos < vals.size() && vals[pos] == "multigpu")
        {
            ++pos;
            multiGPU = std::stoull(vals[pos++]);
        }

        auto_allocate = fft_auto_allocation_default; // default if unspecified
        if(pos < vals.size() && vals[pos] == "autoallocation")
        {
            ++pos;
            lexical_cast(vals[pos++], auto_allocate);
        }
    }

    // Stream output operator (for gtest, etc).
    friend std::ostream& operator<<(std::ostream& stream, const fft_params& params)
    {
        stream << params.str();
        return stream;
    }

    // Dimension of the transform.
    size_t dim() const
    {
        return length.size();
    }

    virtual std::vector<size_t> ilength() const
    {
        auto ilength = length;
        if(transform_type == fft_transform_type_real_inverse)
            ilength[dim() - 1] = ilength[dim() - 1] / 2 + 1;
        return ilength;
    }

    virtual std::vector<size_t> olength() const
    {
        auto olength = length;
        if(transform_type == fft_transform_type_real_forward)
            olength[dim() - 1] = olength[dim() - 1] / 2 + 1;
        return olength;
    }

    static size_t nbuffer(const fft_array_type type)
    {
        switch(type)
        {
        case fft_array_type_real:
        case fft_array_type_complex_interleaved:
        case fft_array_type_hermitian_interleaved:
            return 1;
        case fft_array_type_complex_planar:
        case fft_array_type_hermitian_planar:
            return 2;
        case fft_array_type_unset:
            return 0;
        }
        return 0;
    }

    // Number of input buffers
    size_t nibuffer() const
    {
        return nbuffer(itype);
    }

    // Number of output buffers
    size_t nobuffer() const
    {
        return nbuffer(otype);
    }

    void set_iotypes()
    {
        if(itype == fft_array_type_unset)
        {
            switch(transform_type)
            {
            case fft_transform_type_complex_forward:
            case fft_transform_type_complex_inverse:
                itype = fft_array_type_complex_interleaved;
                break;
            case fft_transform_type_real_forward:
                itype = fft_array_type_real;
                break;
            case fft_transform_type_real_inverse:
                itype = fft_array_type_hermitian_interleaved;
                break;
            default:
                throw std::runtime_error("Invalid transform type");
            }
        }
        if(otype == fft_array_type_unset)
        {
            switch(transform_type)
            {
            case fft_transform_type_complex_forward:
            case fft_transform_type_complex_inverse:
                otype = fft_array_type_complex_interleaved;
                break;
            case fft_transform_type_real_forward:
                otype = fft_array_type_hermitian_interleaved;
                break;
            case fft_transform_type_real_inverse:
                otype = fft_array_type_real;
                break;
            default:
                throw std::runtime_error("Invalid transform type");
            }
        }
    }

    // Check that the input and output types are consistent.
    bool check_iotypes() const
    {
        switch(itype)
        {
        case fft_array_type_complex_interleaved:
        case fft_array_type_complex_planar:
        case fft_array_type_hermitian_interleaved:
        case fft_array_type_hermitian_planar:
        case fft_array_type_real:
            break;
        default:
            throw std::runtime_error("Invalid Input array type format");
        }

        switch(otype)
        {
        case fft_array_type_complex_interleaved:
        case fft_array_type_complex_planar:
        case fft_array_type_hermitian_interleaved:
        case fft_array_type_hermitian_planar:
        case fft_array_type_real:
            break;
        default:
            throw std::runtime_error("Invalid Input array type format");
        }

        // Check that format choices are supported
        if(transform_type != fft_transform_type_real_forward
           && transform_type != fft_transform_type_real_inverse)
        {
            if(placement == fft_placement_inplace && itype != otype)
            {
                throw std::runtime_error(
                    "In-place transforms must have identical input and output types");
            }
        }

        bool okformat = true;
        switch(itype)
        {
        case fft_array_type_complex_interleaved:
        case fft_array_type_complex_planar:
            okformat = (otype == fft_array_type_complex_interleaved
                        || otype == fft_array_type_complex_planar);
            break;
        case fft_array_type_hermitian_interleaved:
        case fft_array_type_hermitian_planar:
            okformat = otype == fft_array_type_real;
            break;
        case fft_array_type_real:
            okformat = (otype == fft_array_type_hermitian_interleaved
                        || otype == fft_array_type_hermitian_planar);
            break;
        default:
            throw std::runtime_error("Invalid Input array type format");
        }

        return okformat;
    }

    // Given a length vector, set the rest of the strides.
    // The optional argument stride0 sets the stride for the contiguous dimension.
    // The optional rcpadding argument sets the stride correctly for in-place
    // multi-dimensional real/complex transforms.
    // Format is row-major.
    template <typename T1>
    std::vector<T1> compute_stride(const std::vector<T1>&     length,
                                   const std::vector<size_t>& stride0   = std::vector<size_t>(),
                                   const bool                 rcpadding = false) const
    {
        std::vector<T1> stride(dim());

        size_t dimoffset = 0;

        if(stride0.size() == 0)
        {
            // Set the contiguous stride:
            stride[dim() - 1] = 1;
            dimoffset         = 1;
        }
        else
        {
            // Copy the input values to the end of the stride array:
            for(size_t i = 0; i < stride0.size(); ++i)
            {
                stride[dim() - stride0.size() + i] = stride0[i];
            }
        }

        if(stride0.size() < dim())
        {
            // Compute any remaining values via recursion.
            for(size_t i = dim() - dimoffset - stride0.size(); i-- > 0;)
            {
                auto lengthip1 = length[i + 1];
                if(rcpadding && i == dim() - 2)
                {
                    lengthip1 = 2 * (lengthip1 / 2 + 1);
                }
                stride[i] = stride[i + 1] * lengthip1;
            }
        }

        return stride;
    }

    void compute_istride()
    {
        istride = compute_stride(ilength(),
                                 istride,
                                 placement == fft_placement_inplace
                                     && transform_type == fft_transform_type_real_forward);
    }

    void compute_ostride()
    {
        ostride = compute_stride(olength(),
                                 ostride,
                                 placement == fft_placement_inplace
                                     && transform_type == fft_transform_type_real_inverse);
    }

    virtual void compute_isize()
    {
        auto   il  = ilength();
        size_t val = compute_ptrdiff(il, istride, nbatch, idist);
        isize.resize(nibuffer());
        for(unsigned int i = 0; i < isize.size(); ++i)
        {
            isize[i] = val + ioffset[i];
        }
    }

    virtual void compute_osize()
    {
        auto   ol  = olength();
        size_t val = compute_ptrdiff(ol, ostride, nbatch, odist);
        osize.resize(nobuffer());
        for(unsigned int i = 0; i < osize.size(); ++i)
        {
            osize[i] = val + ooffset[i];
        }
    }

    std::vector<size_t> ibuffer_sizes() const
    {
        std::vector<size_t> ibuffer_sizes;

        // In-place real-to-complex transforms need to have enough space in the input buffer to
        // accommodate the output, which is slightly larger.
        if(placement == fft_placement_inplace && transform_type == fft_transform_type_real_forward)
        {
            return obuffer_sizes();
        }

        if(isize.empty())
            return ibuffer_sizes;

        switch(itype)
        {
        case fft_array_type_complex_planar:
        case fft_array_type_hermitian_planar:
            ibuffer_sizes.resize(2);
            break;
        default:
            ibuffer_sizes.resize(1);
        }
        for(unsigned i = 0; i < ibuffer_sizes.size(); i++)
        {
            ibuffer_sizes[i] = isize[i] * var_size<size_t>(precision, itype);
        }
        return ibuffer_sizes;
    }

    virtual std::vector<size_t> obuffer_sizes() const
    {
        std::vector<size_t> obuffer_sizes;

        if(osize.empty())
            return obuffer_sizes;

        switch(otype)
        {
        case fft_array_type_complex_planar:
        case fft_array_type_hermitian_planar:
            obuffer_sizes.resize(2);
            break;
        default:
            obuffer_sizes.resize(1);
        }
        for(unsigned i = 0; i < obuffer_sizes.size(); i++)
        {
            obuffer_sizes[i] = osize[i] * var_size<size_t>(precision, otype);
        }
        return obuffer_sizes;
    }

    // Compute the idist for a given transform based on the placeness, transform type, and data
    // layout.
    size_t compute_idist() const
    {
        size_t dist = 0;
        // In-place 1D transforms need extra dist.
        if(transform_type == fft_transform_type_real_forward && dim() == 1
           && placement == fft_placement_inplace)
        {
            dist = 2 * (length[0] / 2 + 1) * istride[0];
            return dist;
        }

        if(transform_type == fft_transform_type_real_inverse && dim() == 1)
        {
            dist = (length[0] / 2 + 1) * istride[0];
            return dist;
        }

        dist = (transform_type == fft_transform_type_real_inverse)
                   ? (length[dim() - 1] / 2 + 1) * istride[dim() - 1]
                   : length[dim() - 1] * istride[dim() - 1];
        for(unsigned int i = 0; i < dim() - 1; ++i)
        {
            dist = std::max(length[i] * istride[i], dist);
        }
        return dist;
    }
    void set_idist()
    {
        if(idist != 0)
            return;
        idist = compute_idist();
    }

    // Compute the odist for a given transform based on the placeness, transform type, and data
    // layout.  Row-major.
    size_t compute_odist() const
    {
        size_t dist = 0;
        // In-place 1D transforms need extra dist.
        if(transform_type == fft_transform_type_real_inverse && dim() == 1
           && placement == fft_placement_inplace)
        {
            dist = 2 * (length[0] / 2 + 1) * ostride[0];
            return dist;
        }

        if(transform_type == fft_transform_type_real_forward && dim() == 1)
        {
            dist = (length[0] / 2 + 1) * ostride[0];
            return dist;
        }

        dist = (transform_type == fft_transform_type_real_forward)
                   ? (length[dim() - 1] / 2 + 1) * ostride[dim() - 1]
                   : length[dim() - 1] * ostride[dim() - 1];
        for(unsigned int i = 0; i < dim() - 1; ++i)
        {
            dist = std::max(length[i] * ostride[i], dist);
        }
        return dist;
    }
    void set_odist()
    {
        if(odist != 0)
            return;
        odist = compute_odist();
    }

    // Put the length, stride, batch, and dist into a single length/stride array and pass off to the
    // validity checker.
    bool valid_length_stride_batch_dist(const std::vector<size_t>& l0,
                                        const std::vector<size_t>& s0,
                                        const size_t               n,
                                        const size_t               dist,
                                        const int                  verbose = 0) const
    {
        if(l0.size() != s0.size())
            return false;

        // Length and stride vectors, including bathes:
        std::vector<size_t> l{}, s{};
        for(unsigned int i = 0; i < l0.size(); ++i)
        {
            if(l0[i] > 1)
            {
                if(s0[i] == 0)
                    return false;
                l.push_back(l0[i]);
                s.push_back(s0[i]);
            }
        }
        if(n > 1)
        {
            if(dist == 0)
                return false;
            l.push_back(n);
            s.push_back(dist);
        }

        return array_valid(l, s, verbose);
    }

    // Return true if the given GPU parameters would produce a valid transform.
    bool valid(const int verbose = 0) const
    {
        if(ioffset.size() < nibuffer() || ooffset.size() < nobuffer())
            return false;

        // Check that in-place transforms have the same input and output stride:
        if(placement == fft_placement_inplace)
        {
            const auto stridesize = std::min(istride.size(), ostride.size());
            bool       samestride = true;
            for(unsigned int i = 0; i < stridesize; ++i)
            {
                // (strides are irrelevant for unit lengths)
                if(istride[i] != ostride[i] && length[i] > 1)
                    samestride = false;
            }
            if((transform_type == fft_transform_type_complex_forward
                || transform_type == fft_transform_type_complex_inverse)
               && !samestride)
            {
                // In-place transforms require identical input and output strides.
                if(verbose)
                {
                    std::cout << "istride:";
                    for(const auto& i : istride)
                        std::cout << " " << i;
                    std::cout << " ostride:";
                    for(const auto& i : ostride)
                        std::cout << " " << i;
                    std::cout << " differ; skipped for in-place transforms: skipping test"
                              << std::endl;
                }
                return false;
            }

            if((transform_type == fft_transform_type_complex_forward
                || transform_type == fft_transform_type_complex_inverse)
               && (idist != odist) && nbatch > 1)
            {
                // In-place transforms require identical distance, if
                // batch > 1.  If batch is 1 then dist is ignored and
                // the FFT should still work.
                if(verbose)
                {
                    std::cout << "idist:" << idist << " odist:" << odist
                              << " differ; skipped for in-place transforms: skipping test"
                              << std::endl;
                }
                return false;
            }

            if(transform_type == fft_transform_type_real_forward
               || transform_type == fft_transform_type_real_inverse)
            {
                bool invalid = length.back() > 1 && (istride.back() != 1 || ostride.back() != 1);
                if(invalid)
                {
                    // In-place real/complex transforms require unit strides.
                    if(verbose)
                    {
                        std::cout << "istride.back(): " << istride.back()
                                  << " ostride.back(): " << ostride.back()
                                  << " must be unitary for in-place real/complex transforms: "
                                     "skipping test"
                                  << std::endl;
                    }
                    return false;
                }
                const auto& real_strides
                    = transform_type == fft_transform_type_real_forward ? istride : ostride;
                const auto& hermitian_strides
                    = transform_type == fft_transform_type_real_forward ? ostride : istride;
                const auto& real_dist
                    = transform_type == fft_transform_type_real_forward ? idist : odist;
                const auto& hermitian_dist
                    = transform_type == fft_transform_type_real_forward ? odist : idist;
                for(size_t dim = 0; dim < stridesize - 1; dim++)
                {
                    if(length[dim] == 1)
                        continue;
                    invalid |= real_strides[dim] != 2 * hermitian_strides[dim];
                }
                if(nbatch > 1)
                    invalid |= real_dist != 2 * hermitian_dist;
                if(invalid)
                {
                    if(verbose)
                    {
                        std::cout << "Inconsistency detected in strides/distances for in-place "
                                     "real/complex transforms; skipped\n";
                    }
                    return false;
                }
            }

            if((itype == fft_array_type_complex_interleaved
                && otype == fft_array_type_complex_planar)
               || (itype == fft_array_type_complex_planar
                   && otype == fft_array_type_complex_interleaved))
            {
                if(verbose)
                {
                    std::cout << "In-place c2c transforms require identical io types; skipped.\n";
                }
                return false;
            }

            // Check offsets
            switch(transform_type)
            {
            case fft_transform_type_complex_forward:
            case fft_transform_type_complex_inverse:
                for(unsigned int i = 0; i < nibuffer(); ++i)
                {
                    if(ioffset[i] != ooffset[i])
                        return false;
                }
                break;
            case fft_transform_type_real_forward:
                if(ioffset[0] != 2 * ooffset[0])
                    return false;
                break;
            case fft_transform_type_real_inverse:
                if(2 * ioffset[0] != ooffset[0])
                    return false;
                break;
            }
        }

        if(!check_iotypes())
            return false;

        // we can only check output strides on out-of-place
        // transforms, since we need to initialize output to a known
        // pattern
        if(placement == fft_placement_inplace && check_output_strides)
            return false;

        // Check input and output strides
        if(valid_length_stride_batch_dist(ilength(), istride, nbatch, idist, verbose) != true)
        {
            if(verbose)
                std::cout << "Invalid input data format.\n";
            return false;
        }
        if(!(ilength() == olength() && istride == ostride && idist == odist))
        {
            // Only check if different
            if(valid_length_stride_batch_dist(olength(), ostride, nbatch, odist, verbose) != true)
            {
                if(verbose)
                    std::cout << "Invalid output data format.\n";
                return false;
            }
        }

        // The parameters are valid.
        return true;
    }

    // Fill in any missing parameters.
    void validate()
    {
        set_iotypes();
        compute_istride();
        compute_ostride();
        set_idist();
        set_odist();
        compute_isize();
        compute_osize();

        validate_fields();
    }

    // validate that the bricks in the fields have positive volume
    // (i.e. upper index is above lower index).  also check that they
    // have the right number of dimensions for the fields
    void validate_brick_volume() const
    {
        // row-major lengths including batch (i.e. batch is at the front)
        std::vector<size_t> length_with_batch{nbatch};
        std::copy(length.begin(), length.end(), std::back_inserter(length_with_batch));

        auto validate_field = [&](const fft_field& f) {
            for(const auto& b : f.bricks)
            {
                // bricks must have same dim as FFT, including batch
                if(b.lower.size() != length.size() + 1 || b.upper.size() != length.size() + 1
                   || b.stride.size() != length.size() + 1)
                    throw std::runtime_error(
                        "brick dimension does not match FFT + batch dimension");

                // ensure lower < upper, and that both fit in the FFT + batch dims
                if(!std::lexicographical_compare(
                       b.lower.begin(), b.lower.end(), b.upper.begin(), b.upper.end()))
                    throw std::runtime_error("brick lower index is not less than upper index");

                if(!std::lexicographical_compare(b.lower.begin(),
                                                 b.lower.end(),
                                                 length_with_batch.begin(),
                                                 length_with_batch.end()))
                    throw std::runtime_error(
                        "brick lower index is not less than FFT + batch length");

                if(!std::lexicographical_compare(b.upper.begin(),
                                                 b.upper.end(),
                                                 length_with_batch.begin(),
                                                 length_with_batch.end())
                   && b.upper != length_with_batch)
                    throw std::runtime_error("brick upper index is not <= FFT + batch length");
            }
        };

        for(const auto& ifield : ifields)
            validate_field(ifield);
        for(const auto& ofield : ofields)
            validate_field(ofield);
    }

    virtual void validate_fields() const
    {
        if(multiGPU > 1)
            throw std::runtime_error("library-decomposed multi-GPU is unsupported");
    }

    // Column-major getters:
    std::vector<size_t> length_cm() const
    {
        auto length_cm = length;
        std::reverse(std::begin(length_cm), std::end(length_cm));
        return length_cm;
    }
    std::vector<size_t> ilength_cm() const
    {
        auto ilength_cm = ilength();
        std::reverse(std::begin(ilength_cm), std::end(ilength_cm));
        return ilength_cm;
    }
    std::vector<size_t> olength_cm() const
    {
        auto olength_cm = olength();
        std::reverse(std::begin(olength_cm), std::end(olength_cm));
        return olength_cm;
    }
    std::vector<size_t> istride_cm() const
    {
        auto istride_cm = istride;
        std::reverse(std::begin(istride_cm), std::end(istride_cm));
        return istride_cm;
    }
    std::vector<size_t> ostride_cm() const
    {
        auto ostride_cm = ostride;
        std::reverse(std::begin(ostride_cm), std::end(ostride_cm));
        return ostride_cm;
    }
    bool is_interleaved() const
    {
        if(itype == fft_array_type_complex_interleaved
           || itype == fft_array_type_hermitian_interleaved)
            return true;
        if(otype == fft_array_type_complex_interleaved
           || otype == fft_array_type_hermitian_interleaved)
            return true;
        return false;
    }
    bool is_planar() const
    {
        if(itype == fft_array_type_complex_planar || itype == fft_array_type_hermitian_planar)
            return true;
        if(otype == fft_array_type_complex_planar || otype == fft_array_type_hermitian_planar)
            return true;
        return false;
    }
    bool is_real() const
    {
        return (itype == fft_array_type_real || otype == fft_array_type_real);
    }
    bool is_callback() const
    {
        return run_callbacks;
    }
    // checks if the parameters are consistent with a "default" data layout (considering strides and distances)
    bool is_using_default_layout() const
    {
        static_assert(std::is_same_v<decltype(ioffset), decltype(ooffset)>);
        auto is_zero = [](const decltype(ioffset)::value_type& i) { return i == 0; };
        return std::all_of(ioffset.begin(), ioffset.end(), is_zero)
               && std::all_of(ooffset.begin(), ooffset.end(), is_zero)
               && istride == default_strides(transform_type, placement, fft_io::fft_io_in, length)
               && ostride == default_strides(transform_type, placement, fft_io::fft_io_out, length)
               && idist
                      == default_distance(
                          transform_type, placement, fft_io::fft_io_in, length, nbatch)
               && odist
                      == default_distance(
                          transform_type, placement, fft_io::fft_io_out, length, nbatch);
    }

    // Given a data type and dimensions, fill the buffer, imposing Hermitian symmetry if necessary.
    template <typename Tbuff>
    inline void compute_input(std::vector<Tbuff>& input)
    {
        auto deviceProp = get_curr_device_prop();

        std::vector<size_t> field_lower(dim());
        auto                contiguous_stride = compute_stride(ilength());
        auto                contiguous_dist   = compute_idist();

        switch(precision)
        {
        case fft_precision_half:
            set_input<Tbuff, rocfft_fp16>(input,
                                          igen,
                                          itype,
                                          length,
                                          ilength(),
                                          istride,
                                          ioffset,
                                          idist,
                                          nbatch,
                                          deviceProp,
                                          field_lower,
                                          0,
                                          contiguous_stride,
                                          contiguous_dist);
            break;
        case fft_precision_double:
            set_input<Tbuff, double>(input,
                                     igen,
                                     itype,
                                     length,
                                     ilength(),
                                     istride,
                                     ioffset,
                                     idist,
                                     nbatch,
                                     deviceProp,
                                     field_lower,
                                     0,
                                     contiguous_stride,
                                     contiguous_dist);
            break;
        case fft_precision_single:
            set_input<Tbuff, float>(input,
                                    igen,
                                    itype,
                                    length,
                                    ilength(),
                                    istride,
                                    ioffset,
                                    idist,
                                    nbatch,
                                    deviceProp,
                                    field_lower,
                                    0,
                                    contiguous_stride,
                                    contiguous_dist);
            break;
        }
    }

    template <typename Tstream = std::ostream>
    void print_ibuffer(const std::vector<hostbuf>& buf, Tstream& stream = std::cout) const
    {
        switch(itype)
        {
        case fft_array_type_complex_interleaved:
        case fft_array_type_hermitian_interleaved:
        {
            switch(precision)
            {
            case fft_precision_half:
            {
                buffer_printer<rocfft_complex<rocfft_fp16>> s;
                s.print_buffer_half(buf, ilength(), istride, nbatch, idist, ioffset);
                break;
            }
            case fft_precision_single:
            {
                buffer_printer<rocfft_complex<float>> s;
                s.print_buffer_single(buf, ilength(), istride, nbatch, idist, ioffset);
                break;
            }
            case fft_precision_double:
            {
                buffer_printer<rocfft_complex<double>> s;
                s.print_buffer_double(buf, ilength(), istride, nbatch, idist, ioffset);
                break;
            }
            }
            break;
        }
        case fft_array_type_complex_planar:
        case fft_array_type_hermitian_planar:
        case fft_array_type_real:
        {
            switch(precision)
            {
            case fft_precision_half:
            {
                buffer_printer<rocfft_fp16> s;
                s.print_buffer_half(buf, ilength(), istride, nbatch, idist, ioffset);
                break;
            }
            case fft_precision_single:
            {
                buffer_printer<float> s;
                s.print_buffer_single(buf, ilength(), istride, nbatch, idist, ioffset);
                break;
            }
            case fft_precision_double:
            {
                buffer_printer<double> s;
                s.print_buffer_double(buf, ilength(), istride, nbatch, idist, ioffset);
                break;
            }
            }
            break;
        }
        default:
            throw std::runtime_error("Invalid itype in print_ibuffer");
        }
    }

    template <typename Tstream = std::ostream>
    void print_obuffer(const std::vector<hostbuf>& buf, Tstream& stream = std::cout) const
    {
        switch(otype)
        {
        case fft_array_type_complex_interleaved:
        case fft_array_type_hermitian_interleaved:
        {
            switch(precision)
            {
            case fft_precision_half:
            {
                buffer_printer<rocfft_complex<rocfft_fp16>> s;
                s.print_buffer_half(buf, olength(), ostride, nbatch, odist, ooffset);
                break;
            }
            case fft_precision_single:
            {
                buffer_printer<rocfft_complex<float>> s;
                s.print_buffer_single(buf, olength(), ostride, nbatch, odist, ooffset);
                break;
            }
            case fft_precision_double:
                buffer_printer<rocfft_complex<double>> s;
                s.print_buffer_double(buf, olength(), ostride, nbatch, odist, ooffset);
                break;
            }
            break;
        }
        case fft_array_type_complex_planar:
        case fft_array_type_hermitian_planar:
        case fft_array_type_real:
        {
            switch(precision)
            {
            case fft_precision_half:
            {
                buffer_printer<rocfft_fp16> s;
                s.print_buffer_half(buf, olength(), ostride, nbatch, odist, ooffset);
                break;
            }
            case fft_precision_single:
            {
                buffer_printer<float> s;
                s.print_buffer_single(buf, olength(), ostride, nbatch, odist, ooffset);
                break;
            }
            case fft_precision_double:
            {
                buffer_printer<double> s;
                s.print_buffer_double(buf, olength(), ostride, nbatch, odist, ooffset);
                break;
            }
            }
            break;
        }

        default:
            throw std::runtime_error("Invalid itype in print_obuffer");
        }
    }

    void print_ibuffer_flat(const std::vector<hostbuf>& buf) const
    {
        switch(itype)
        {
        case fft_array_type_complex_interleaved:
        case fft_array_type_hermitian_interleaved:
        {
            switch(precision)
            {
            case fft_precision_half:
            {
                buffer_printer<rocfft_complex<rocfft_fp16>> s;
                s.print_buffer_flat(buf, osize, ooffset);
                break;
            }
            case fft_precision_single:
            {
                buffer_printer<rocfft_complex<float>> s;
                s.print_buffer_flat(buf, osize, ooffset);
                break;
            }
            case fft_precision_double:
                buffer_printer<rocfft_complex<double>> s;
                s.print_buffer_flat(buf, osize, ooffset);
                break;
            }
            break;
        }
        case fft_array_type_complex_planar:
        case fft_array_type_hermitian_planar:
        case fft_array_type_real:
        {
            switch(precision)
            {
            case fft_precision_half:
            {
                buffer_printer<rocfft_fp16> s;
                s.print_buffer_flat(buf, osize, ooffset);
                break;
            }
            case fft_precision_single:
            {
                buffer_printer<float> s;
                s.print_buffer_flat(buf, osize, ooffset);
                break;
            }
            case fft_precision_double:
            {
                buffer_printer<double> s;
                s.print_buffer_flat(buf, osize, ooffset);
                break;
            }
            }
            break;
        default:
            throw std::runtime_error("Invalid itype in print_ibuffer_flat");
        }
        }
    }

    void print_obuffer_flat(const std::vector<hostbuf>& buf) const
    {
        switch(otype)
        {
        case fft_array_type_complex_interleaved:
        case fft_array_type_hermitian_interleaved:
        {
            switch(precision)
            {
            case fft_precision_half:
            {
                buffer_printer<rocfft_complex<rocfft_fp16>> s;
                s.print_buffer_flat(buf, osize, ooffset);
                break;
            }
            case fft_precision_single:
            {
                buffer_printer<rocfft_complex<float>> s;
                s.print_buffer_flat(buf, osize, ooffset);
                break;
            }
            case fft_precision_double:
                buffer_printer<rocfft_complex<double>> s;
                s.print_buffer_flat(buf, osize, ooffset);
                break;
            }
            break;
        }
        case fft_array_type_complex_planar:
        case fft_array_type_hermitian_planar:
        case fft_array_type_real:
        {
            switch(precision)
            {
            case fft_precision_half:
            {
                buffer_printer<rocfft_fp16> s;
                s.print_buffer_flat(buf, osize, ooffset);
                break;
            }
            case fft_precision_single:
            {
                buffer_printer<float> s;
                s.print_buffer_flat(buf, osize, ooffset);
                break;
            }

            case fft_precision_double:
            {
                buffer_printer<double> s;
                s.print_buffer_flat(buf, osize, ooffset);
                break;
            }
            }
            break;
        default:
            throw std::runtime_error("Invalid itype in print_ibuffer_flat");
        }
        }
    }

    // A callback is expressed as a pair of device function pointer +
    // device function data.
    //
    // Load and store callbacks are provided as vectors of those
    // pointers, as we need a separate function+data for each device
    // being loaded from or stored to.
    virtual fft_status set_callbacks(std::vector<void*>* load_cb_func,
                                     std::vector<void*>* load_cb_data,
                                     std::vector<void*>* store_cb_func,
                                     std::vector<void*>* store_cb_data,
                                     size_t              load_cb_shared_mem_bytes,
                                     size_t              store_cb_shared_mem_bytes)
    {
        return fft_status_success;
    }

    virtual fft_status execute(void** in, void** out)
    {
        return fft_status_success;
    }

    std::vector<size_t> footprint_on_devices_for(fft_io io) const
    {
        std::vector<size_t> sizes(rocfft_scoped_device::device_count());
        const auto&         iofields = io == fft_io::fft_io_in ? ifields : ofields;
        const auto          iobuffer_size
            = io == fft_io::fft_io_in ? sum(ibuffer_sizes()) : sum(obuffer_sizes());

        // If this is library-decomposed multi-GPU, only the library
        // can really say what the footprint will be.  Estimate it
        // here by assuming the input/output size will be evenly
        // divided across all of the devices.
        if(multiGPU > 1)
        {
            // We will use the first N devices for library-decomposed
            for(size_t device = 0; device < multiGPU; ++device)
            {
                sizes.at(device) += DivRoundingUp(iobuffer_size, multiGPU);
            }
            return sizes;
        }

        int currentDevice = hipInvalidDeviceId;
        if(hipGetDevice(&currentDevice) != hipSuccess)
            throw std::runtime_error("hipGetDevice failed");

        // add sizes for field if specified, otherwise assume
        // single-device input/output buffer on current device
        if(iofields.empty())
        {
            sizes.at(currentDevice) += iobuffer_size;
        }
        else
        {
            // add footprint of each brick to its device
            for(const auto& field : iofields)
            {
                for(const auto& brick : field.bricks)
                {
                    sizes.at(brick.device) += compute_ptrdiff(brick.length(), brick.stride);
                }
            }
        }
        return sizes;
    }

    // return the per-device memory footprint of just the input/output data
    std::vector<size_t> io_vram_footprint() const
    {
        std::vector<size_t> sizes(rocfft_scoped_device::device_count());
        const auto          input_footprints  = footprint_on_devices_for(fft_io::fft_io_in);
        const auto          output_footprints = footprint_on_devices_for(fft_io::fft_io_out);

        for(size_t i = 0; i < sizes.size(); ++i)
        {
            if(placement == fft_placement_inplace)
            {
                // inplace means we need the maximum of input and output for each device
                sizes.at(i) = std::max(input_footprints.at(i), output_footprints.at(i));
            }
            else
            {
                // out of place means input and output are separate
                sizes.at(i) = input_footprints.at(i) + output_footprints.at(i);
            }
        }
        return sizes;
    }

    // return the full per-device memory footprint of the FFT including
    // any work buffers required by the FFT library
    virtual std::vector<size_t> vram_footprint()
    {
        return io_vram_footprint();
    }

    // Specific exception type for work buffer allocation failure.
    // Tests that hit this can't fit on the GPU and should be skipped.
    struct work_buffer_alloc_failure : public std::runtime_error
    {
        const size_t attempted_size;
        work_buffer_alloc_failure(const std::string& s, size_t _attempted_size = 0)
            : std::runtime_error(s)
            , attempted_size(_attempted_size)
        {
        }
    };

    // Specific exception type for unimplemented feature(s).
    struct unimplemented_exception : public std::runtime_error
    {
        unimplemented_exception(const std::string& s)
            : std::runtime_error(s)
        {
        }
    };

    virtual fft_status create_plan()
    {
        return fft_status_success;
    }

    // Change a forward transform to it's inverse
    void inverse_from_forward(fft_params& params_forward)
    {
        switch(params_forward.transform_type)
        {
        case fft_transform_type_complex_forward:
            transform_type = fft_transform_type_complex_inverse;
            break;
        case fft_transform_type_real_forward:
            transform_type = fft_transform_type_real_inverse;
            break;
        default:
            throw std::runtime_error("Transform type not forward.");
        }

        length        = params_forward.length;
        istride       = params_forward.ostride;
        ostride       = params_forward.istride;
        nbatch        = params_forward.nbatch;
        precision     = params_forward.precision;
        placement     = params_forward.placement;
        idist         = params_forward.odist;
        odist         = params_forward.idist;
        itype         = params_forward.otype;
        otype         = params_forward.itype;
        ioffset       = params_forward.ooffset;
        ooffset       = params_forward.ioffset;
        auto_allocate = params_forward.auto_allocate;

        run_callbacks = params_forward.run_callbacks;
        multiGPU      = params_forward.multiGPU;

        check_output_strides = params_forward.check_output_strides;

        scale_factor = 1 / params_forward.scale_factor;
    }

    /**
     * @brief initialize the I/O device buffers as required for multi-device
     * transforms described by this object. Device buffers are determined (created
     * and allocated, if needed), and input buffers are data-initialized using the
     * provided host-residing (resp. device-residing) reference data for the
     * `rocfft_params` (resp. `hipfft_params`) derived class.
     *
     * TODO: unify data-initialization logic across derived classes and make all arguments
     * relevant in all cases (remove ignore ones).
     * 
     * @param[in] input_data_host vector of (at least) one host buffer containing the
     * input-initialization data to be considered for data-initialization of the
     * device input buffers (this is unused by `hipfft_params` objects). If used,
     * the data layout of `input_data_host[0]` must be consistent with `cpu_params.istride`,
     * `cpu_params.idist`, etc. wherein `cpu_params = this->make_params_for_reference_cpu()`.
     * @param[in] input_data_gpu vector of (at least) one device buffer containing the
     * input-initialization data to be considered for data-initialization of the
     * device input buffers (this is unused by `rocfft_params` objects). If used,
     * the data layout of `input_data_gpu[0]` must be consistent with `this->istride`,
     * `this->idist`, etc.
     * @param[out] mgpu_ibuffers vector of raw pointers to input device allocations as
     * needed for the multi-device transform that this object describes. The content
     * of these device allocations is initialized with the (transposed) content of
     * `input_data_host` (resp. `input_data_gpu`) by `rocfft_params` (resp. `hipfft_params`)
     * objects. Upon return, the data layout in these device allocations is
     * - as described by the corresponding input brick's for `rocfft_params` objects;
     * - as implicitly-defined for `hipfft_params` objects.
     * @param[out] mgpu_obuffers vector of raw pointers to output device allocations as
     * needed for the multi-device transform that this object describes.
     */
    virtual void multi_gpu_prepare(const std::vector<hostbuf>& input_data_host,
                                   const std::vector<gpubuf>&  input_data_gpu,
                                   std::vector<void*>&         mgpu_ibuffers,
                                   std::vector<void*>&         mgpu_obuffers)
    {
    }

    /**
     * @brief gather the results of a multi-device transform to a unique host-residing
     * buffer for `rocfft_params` objects (resp. device-residing buffer for
     * `hipfft_params` objects).
     * 
     * TODO: unify data-gathering logic across derived classes and make all arguments
     * relevant in all cases (remove ignore ones).
     * 
     * @param[out] gathered_results_host vector of (at least) one host buffer (ignored by
     * `hipfft_params` objects). If not ignored, `gathered_results_host[0]` contains the
     * gathered output data of the multi-device transform upon return.
     * @param[out] gathered_results_device vector of (at least) one device buffer (ignored
     * by `rocfft_params` objects). If not ignored, `gathered_results_device[0]` contains
     * the gathered output data of the multi-device transform upon return.
     * @param[in] mgpu_obuffers vector of raw pointers to output device allocations
     * considered by the (previously-executed) multi-device transform described by this
     * object.
     * 
     * Note 1: whichever gathering buffer that is considered (`gathered_results_host[0]`
     * or `gathered_results_device[0]`) must be large enough for containing the
     * gathered results when passed to this function: this function does NOT resize it.
     * Note 2: the data layout in the considered gathered results is defined
     * by the calling object's output data layout parameter, i.e., defined by
     * `this->ostride`, `this->odist`, `this->otype`, etc.
     */
    virtual void multi_gpu_finalize(std::vector<hostbuf>& gathered_results_host,
                                    std::vector<gpubuf>&  gathered_results_device,
                                    std::vector<void*>&   mgpu_obuffers)
    {
    }

    /**
     * @brief create one input (resp. output) field with as many bricks
     * as desired along every field dimension for the current object.
     * The device assignment of the bricks is cycling though the total number
     * of available devices (i.e., `gpusperrank * num_ranks`) starting from
     * the `start_global_dev_idx` (global) device index.
     * NOTE: The global device index `g` in `[0, gpusperrank * mp_ranks(`
     * corresponds to device of local ID `g / mp_ranks` on process of rank
     * `g % mp_ranks`. This prioritize distributing layout across all processes.
     * 
     * @tparam io enum flag specifying whether the input (`io == fft_io::fft_io_in`)
     * or output (`io == fft_io::fft_io_in`) field is being considered.
     * 
     * @param[in] gpusperrank number of GPU devices available to each process
     * @param[in] brick_count_along desired (strictly positive) numbers of
     * bricks per dimension: `brick_count_along[i]` if the number of bricks
     * desired along the `i`-th field dimension.
     * @param[in] num_ranks number of processes used in the parallel computer
     * (default value is 1)
     * @param[in] start_global_dev_idx first (global) device ID to be considered
     * in the cyclic assignment (default value is 0)
     * 
     * NOTES:
     * - data layouts within bricks are set to be compact, i.e., contiguous for
     * bricks in complex domain, padded in real domain for in-place operations
     * (only if the fastest dimension is not divided itself, i.e., if
     * brick_count_along.last() == 1)
     * - if more bricks than available devices are requested, some bricks may be
     * assigned to the same device.
     */
    template <fft_io io>
    void distribute_field(int                              gpusperrank,
                          const std::vector<unsigned int>& brick_count_along,
                          int                              num_ranks            = 1,
                          int                              start_global_dev_idx = 0)
    {
        static_assert(io == fft_io::fft_io_in || io == fft_io::fft_io_out);

        auto& iofields       = io == fft_io::fft_io_in ? ifields : ofields;
        auto  iofield_length = io == fft_io::fft_io_in ? ilength() : olength();
        iofield_length.insert(iofield_length.begin(), nbatch);
        const auto field_size = iofield_length.size(); // --> field_size > 0
        // arg validation
        if(brick_count_along.size() != field_size)
            throw std::runtime_error(
                "fft_params::distribute_field inconsistent size between desired number of bricks "
                "per dimension and number of dimensions");
        if(std::any_of(
               brick_count_along.begin(),
               brick_count_along.end(),
               [](const auto& brick_count_along_dim) { return brick_count_along_dim == 0; }))
            throw std::invalid_argument("brick_count_per_dim must not be 0.");
        if(gpusperrank < 1 || num_ranks < 1)
            throw std::invalid_argument(
                "fft_params::distribute_field requires strictly positive number of processes and "
                "number of available device(s) per process.");

        const auto total_bricks = product(brick_count_along.begin(), brick_count_along.end());

        if(total_bricks == 1 && mp_lib == fft_mp_lib_none && num_ranks == 1
           && start_global_dev_idx == 0)
        {
            // Specifying a unique field with a lone brick on the current device may
            // be omitted in favor of plan's data layout parameters: test that by not
            // using a lone-brick field sometimes, in such cases.
            const auto stable_hash_str
                = token() + (io == fft_io::fft_io_in ? "_input_field" : "_output_field");
            if(std::hash<std::string>()(stable_hash_str) % 2 == 1)
                return;
        }

        struct length_division
        {
            size_t min_length;
            size_t remainder;
        };
        std::vector<length_division> field_division_along(field_size);
        for(auto field_dim = field_size; field_dim-- > 0;)
        {
            field_division_along[field_dim].min_length
                = iofield_length[field_dim] / brick_count_along[field_dim];
            field_division_along[field_dim].remainder
                = iofield_length[field_dim] % brick_count_along[field_dim];
        }

        // make ONE field with as many bricks as required along every field dimension
        iofields.resize(1);
        auto& iofield  = iofields[0];
        auto& iobricks = iofield.bricks;
        // Create the required number of bricks
        iobricks.resize(total_bricks);
        // define them
        for(size_t b_idx = 0; b_idx < total_bricks; b_idx++)
        {
            auto&               iobrick = iobricks[b_idx];
            std::vector<size_t> brick_dim_idx(field_size, 0);
            iobrick.lower.resize(field_size);
            iobrick.upper.resize(field_size);
            iobrick.stride.resize(field_size);
            auto tmp_idx = b_idx;
            for(auto field_dim = field_size; field_dim-- > 0;
                tmp_idx /= brick_count_along[field_dim])
            {
                const auto brick_idx_along_dim = tmp_idx % brick_count_along[field_dim];
                const auto brick_length_along_dim
                    = field_division_along[field_dim].min_length
                      + (brick_idx_along_dim < field_division_along[field_dim].remainder ? 1 : 0);
                iobrick.lower[field_dim]
                    = brick_idx_along_dim * field_division_along[field_dim].min_length
                      + std::min(brick_idx_along_dim, field_division_along[field_dim].remainder);
                iobrick.upper[field_dim] = iobrick.lower[field_dim] + brick_length_along_dim;
                if(field_dim == field_size - 1)
                    iobrick.stride[field_dim] = 1; // contiguous along fastest dimension
                else
                {
                    auto stride_multiplier
                        = iobrick.upper[field_dim + 1] - iobrick.lower[field_dim + 1];
                    if(field_dim == field_size - 2 && is_real()
                       && placement == fft_placement_inplace
                       && (is_forward() ^ (io == fft_io::fft_io_out)) /* <-- brick in real domain */
                       && brick_count_along[field_size - 1] == 1)
                    {
                        // real in-place case with fastest dimension that is NOT divided
                        // --> most likely use case is *with* padding
                        stride_multiplier = 2 * (stride_multiplier / 2 + 1);
                    }
                    iobrick.stride[field_dim] = iobrick.stride[field_dim + 1] * stride_multiplier;
                }
            }

            iobrick.rank   = (start_global_dev_idx + b_idx) % num_ranks;
            iobrick.device = ((start_global_dev_idx + b_idx) / num_ranks) % gpusperrank;
        }
    }

    // Apply load operations specified by this struct to the host-side
    // input before we pass it to the reference FFT
    void apply_host_load_ops(std::vector<hostbuf>& input) const
    {
        // Currently no load operations can be specified
    }

    // Apply store operations specified by this struct to the host-side
    // output after we get it from the reference FFT
    void apply_host_store_ops(std::vector<hostbuf>& output) const
    {
        // Store ops like result scaling are only supported on AMD
        // backend, and CUDA implements some conflicting
        // half-precision operators that prevent result scaling from
        // compiling
#ifdef __HIP_PLATFORM_AMD__
        // Don't bother iterating over the data if we don't have to
        if(scale_factor == 1.0)
            return;
#ifdef _OPENMP
        auto partition_count = compute_partition_count(output.front().size());
#endif

        switch(otype)
        {
            // Reference FFT always works with complex-interleaved data even if
            // params specifies planar
        case fft_array_type_complex_interleaved:
        case fft_array_type_hermitian_interleaved:
        case fft_array_type_complex_planar:
        case fft_array_type_hermitian_planar:
        {
            switch(precision)
            {
            case fft_precision_half:
            {
                const size_t elem_size = sizeof(rocfft_complex<rocfft_fp16>);
                const size_t num_elems = output.front().size() / elem_size;

                auto output_begin
                    = reinterpret_cast<rocfft_complex<rocfft_fp16>*>(output.front().data());
#ifdef _OPENMP
#pragma omp parallel for num_threads(partition_count)
#endif
                for(size_t i = 0; i < num_elems; ++i)
                {
                    auto& element = output_begin[i];
                    if(scale_factor != 1.0)
                        element = element * scale_factor;
                }
                break;
            }
            case fft_precision_single:
            {
                const size_t elem_size = sizeof(rocfft_complex<float>);
                const size_t num_elems = output.front().size() / elem_size;

                auto output_begin = reinterpret_cast<rocfft_complex<float>*>(output.front().data());
#ifdef _OPENMP
#pragma omp parallel for num_threads(partition_count)
#endif
                for(size_t i = 0; i < num_elems; ++i)
                {
                    auto& element = output_begin[i];
                    if(scale_factor != 1.0)
                        element = element * scale_factor;
                }
                break;
            }
            case fft_precision_double:
            {
                const size_t elem_size = sizeof(rocfft_complex<double>);
                const size_t num_elems = output.front().size() / elem_size;

                auto output_begin
                    = reinterpret_cast<rocfft_complex<double>*>(output.front().data());
#ifdef _OPENMP
#pragma omp parallel for num_threads(partition_count)
#endif
                for(size_t i = 0; i < num_elems; ++i)
                {
                    auto& element = output_begin[i];
                    if(scale_factor != 1.0)
                        element = element * scale_factor;
                }
                break;
            }
            }
        }
        break;
        case fft_array_type_real:
        {
            switch(precision)
            {
            case fft_precision_half:
            {
                const size_t elem_size = sizeof(rocfft_fp16);
                const size_t num_elems = output.front().size() / elem_size;

                auto output_begin = reinterpret_cast<rocfft_fp16*>(output.front().data());
#ifdef _OPENMP
#pragma omp parallel for num_threads(partition_count)
#endif
                for(size_t i = 0; i < num_elems; ++i)
                {
                    auto& element = output_begin[i];
                    if(scale_factor != 1.0)
                        element = element * scale_factor;
                }
                break;
            }
            case fft_precision_single:
            {
                const size_t elem_size = sizeof(float);
                const size_t num_elems = output.front().size() / elem_size;

                auto output_begin = reinterpret_cast<float*>(output.front().data());
#ifdef _OPENMP
#pragma omp parallel for num_threads(partition_count)
#endif
                for(size_t i = 0; i < num_elems; ++i)
                {
                    auto& element = output_begin[i];
                    if(scale_factor != 1.0)
                        element = element * scale_factor;
                }
                break;
            }
            case fft_precision_double:
            {
                const size_t elem_size = sizeof(double);
                const size_t num_elems = output.front().size() / elem_size;

                auto output_begin = reinterpret_cast<double*>(output.front().data());
#ifdef _OPENMP
#pragma omp parallel for num_threads(partition_count)
#endif
                for(size_t i = 0; i < num_elems; ++i)
                {
                    auto& element = output_begin[i];
                    if(scale_factor != 1.0)
                        element = element * scale_factor;
                }
                break;
            }
            }
        }
        break;
        default:
            // this is FFTW data which should always be interleaved (if complex)
            abort();
        }
#endif
    }

    fft_params make_params_for_reference_cpu() const
    {
        fft_params ret;
        ret.length         = length;
        ret.precision      = precision;
        ret.placement      = fft_placement_notinplace;
        ret.transform_type = transform_type;
        ret.nbatch         = nbatch;
        ret.run_callbacks  = run_callbacks;
        ret.scale_factor   = scale_factor;
        ret.itype          = is_real() ? (is_forward() ? fft_array_type_real
                                                       : fft_array_type_hermitian_interleaved)
                                       : fft_array_type_complex_interleaved;
        ret.otype          = is_real() ? (is_inverse() ? fft_array_type_real
                                                       : fft_array_type_hermitian_interleaved)
                                       : fft_array_type_complex_interleaved;
        ret.istride
            = default_strides(transform_type, fft_placement_notinplace, fft_io::fft_io_in, length);
        ret.ostride
            = default_strides(transform_type, fft_placement_notinplace, fft_io::fft_io_out, length);
        ret.idist = default_distance(
            transform_type, fft_placement_notinplace, fft_io::fft_io_in, length, nbatch);
        ret.odist = default_distance(
            transform_type, fft_placement_notinplace, fft_io::fft_io_out, length, nbatch);

        ret.validate();

        // other ret's members should be irrelevant for cpu reference calculations
        // (default values)
        return ret;
    }
};

// Used for CLI11 parsing of multi-process library enum
static bool lexical_cast(const std::string& word, fft_params::fft_mp_lib& mp_lib)
{
    if(word == "none")
        mp_lib = fft_params::fft_mp_lib_none;
    else if(word == "mpi")
        mp_lib = fft_params::fft_mp_lib_mpi;
    else
        throw std::runtime_error("Invalid multi-process library specified");
    return true;
}

// This is used with CLI11 so that the user can type an integer on the
// command line and we store into an enum variable
template <typename _Elem, typename _Traits>
std::basic_istream<_Elem, _Traits>& operator>>(std::basic_istream<_Elem, _Traits>& stream,
                                               fft_array_type&                     atype)
{
    unsigned tmp;
    stream >> tmp;
    atype = fft_array_type(tmp);
    return stream;
}

// Similarly for transform type
template <typename _Elem, typename _Traits>
std::basic_istream<_Elem, _Traits>& operator>>(std::basic_istream<_Elem, _Traits>& stream,
                                               fft_transform_type&                 ttype)
{
    unsigned tmp;
    stream >> tmp;
    ttype = fft_transform_type(tmp);
    return stream;
}

// Returns pairs of startindex, endindex, for 1D, 2D, 3D lengths
template <typename T1>
std::vector<std::pair<T1, T1>> partition_colmajor(const T1& length)
{
    return partition_base(length, compute_partition_count(length));
}

// Partition on the rightmost part of the tuple, for col-major indexing
template <typename T1>
std::vector<std::pair<std::tuple<T1, T1>, std::tuple<T1, T1>>>
    partition_colmajor(const std::tuple<T1, T1>& length)
{
    auto partitions = partition_base(std::get<1>(length), compute_partition_count(length));
    std::vector<std::pair<std::tuple<T1, T1>, std::tuple<T1, T1>>> ret(partitions.size());
    for(size_t i = 0; i < partitions.size(); ++i)
    {
        std::get<1>(ret[i].first)  = partitions[i].first;
        std::get<0>(ret[i].first)  = 0;
        std::get<1>(ret[i].second) = partitions[i].second;
        std::get<0>(ret[i].second) = std::get<0>(length);
    }
    return ret;
}
template <typename T1>
std::vector<std::pair<std::tuple<T1, T1, T1>, std::tuple<T1, T1, T1>>>
    partition_colmajor(const std::tuple<T1, T1, T1>& length)
{
    auto partitions = partition_base(std::get<2>(length), compute_partition_count(length));
    std::vector<std::pair<std::tuple<T1, T1, T1>, std::tuple<T1, T1, T1>>> ret(partitions.size());
    for(size_t i = 0; i < partitions.size(); ++i)
    {
        std::get<2>(ret[i].first)  = partitions[i].first;
        std::get<1>(ret[i].first)  = 0;
        std::get<0>(ret[i].first)  = 0;
        std::get<2>(ret[i].second) = partitions[i].second;
        std::get<1>(ret[i].second) = std::get<1>(length);
        std::get<0>(ret[i].second) = std::get<0>(length);
    }
    return ret;
}

// Copy data of dimensions length with strides istride and length idist between batches to
// a buffer with strides ostride and length odist between batches.  The input and output
// types are identical.
template <typename Tval, typename Tint1, typename Tint2, typename Tint3>
inline void copy_buffers_1to1(const Tval*                input,
                              Tval*                      output,
                              const Tint1&               whole_length,
                              const size_t               nbatch,
                              const Tint2&               istride,
                              const size_t               idist,
                              const Tint3&               ostride,
                              const size_t               odist,
                              const std::vector<size_t>& ioffset,
                              const std::vector<size_t>& ooffset)
{
    const bool idx_equals_odx = istride == ostride && idist == odist;
    size_t     idx_base       = 0;
    size_t     odx_base       = 0;
    auto       partitions     = partition_rowmajor(whole_length);
    for(size_t b = 0; b < nbatch; b++, idx_base += idist, odx_base += odist)
    {
#ifdef _OPENMP
#pragma omp parallel for num_threads(partitions.size())
#endif
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto idx = compute_index(index, istride, idx_base);
                const auto odx = idx_equals_odx ? idx : compute_index(index, ostride, odx_base);
                output[odx + ooffset[0]] = input[idx + ioffset[0]];
            } while(increment_rowmajor(index, length));
        }
    }
}

// Copy data of dimensions length with strides istride and length idist between batches to
// a buffer with strides ostride and length odist between batches.  The input type is
// planar and the output type is complex interleaved.
template <typename Tval, typename Tint1, typename Tint2, typename Tint3>
inline void copy_buffers_2to1(const Tval*                input0,
                              const Tval*                input1,
                              rocfft_complex<Tval>*      output,
                              const Tint1&               whole_length,
                              const size_t               nbatch,
                              const Tint2&               istride,
                              const size_t               idist,
                              const Tint3&               ostride,
                              const size_t               odist,
                              const std::vector<size_t>& ioffset,
                              const std::vector<size_t>& ooffset)
{
    const bool idx_equals_odx = istride == ostride && idist == odist;
    size_t     idx_base       = 0;
    size_t     odx_base       = 0;
    auto       partitions     = partition_rowmajor(whole_length);
    for(size_t b = 0; b < nbatch; b++, idx_base += idist, odx_base += odist)
    {
#ifdef _OPENMP
#pragma omp parallel for num_threads(partitions.size())
#endif
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto idx = compute_index(index, istride, idx_base);
                const auto odx = idx_equals_odx ? idx : compute_index(index, ostride, odx_base);
                output[odx + ooffset[0]]
                    = rocfft_complex<Tval>(input0[idx + ioffset[0]], input1[idx + ioffset[1]]);
            } while(increment_rowmajor(index, length));
        }
    }
}

// Copy data of dimensions length with strides istride and length idist between batches to
// a buffer with strides ostride and length odist between batches.  The input type is
// complex interleaved and the output type is planar.
template <typename Tval, typename Tint1, typename Tint2, typename Tint3>
inline void copy_buffers_1to2(const rocfft_complex<Tval>* input,
                              Tval*                       output0,
                              Tval*                       output1,
                              const Tint1&                whole_length,
                              const size_t                nbatch,
                              const Tint2&                istride,
                              const size_t                idist,
                              const Tint3&                ostride,
                              const size_t                odist,
                              const std::vector<size_t>&  ioffset,
                              const std::vector<size_t>&  ooffset)
{
    const bool idx_equals_odx = istride == ostride && idist == odist;
    size_t     idx_base       = 0;
    size_t     odx_base       = 0;
    auto       partitions     = partition_rowmajor(whole_length);
    for(size_t b = 0; b < nbatch; b++, idx_base += idist, odx_base += odist)
    {
#ifdef _OPENMP
#pragma omp parallel for num_threads(partitions.size())
#endif
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            auto       index  = partitions[part].first;
            const auto length = partitions[part].second;
            do
            {
                const auto idx = compute_index(index, istride, idx_base);
                const auto odx = idx_equals_odx ? idx : compute_index(index, ostride, odx_base);
                output0[odx + ooffset[0]] = input[idx + ioffset[0]].real();
                output1[odx + ooffset[1]] = input[idx + ioffset[0]].imag();
            } while(increment_rowmajor(index, length));
        }
    }
}

// Copy data of dimensions length with strides istride and length idist between batches to
// a buffer with strides ostride and length odist between batches.  The input type given
// by itype, and the output type is given by otype.
template <typename Tint1, typename Tint2, typename Tint3>
inline void copy_buffers(const std::vector<hostbuf>& input,
                         std::vector<hostbuf>&       output,
                         const Tint1&                length,
                         const size_t                nbatch,
                         const fft_precision         precision,
                         const fft_array_type        itype,
                         const Tint2&                istride,
                         const size_t                idist,
                         const fft_array_type        otype,
                         const Tint3&                ostride,
                         const size_t                odist,
                         const std::vector<size_t>&  ioffset,
                         const std::vector<size_t>&  ooffset)
{
    if(itype == otype)
    {
        switch(itype)
        {
        case fft_array_type_complex_interleaved:
        case fft_array_type_hermitian_interleaved:
            switch(precision)
            {
            case fft_precision_half:
                copy_buffers_1to1(
                    reinterpret_cast<const rocfft_complex<rocfft_fp16>*>(input[0].data()),
                    reinterpret_cast<rocfft_complex<rocfft_fp16>*>(output[0].data()),
                    length,
                    nbatch,
                    istride,
                    idist,
                    ostride,
                    odist,
                    ioffset,
                    ooffset);
                break;
            case fft_precision_single:
                copy_buffers_1to1(reinterpret_cast<const rocfft_complex<float>*>(input[0].data()),
                                  reinterpret_cast<rocfft_complex<float>*>(output[0].data()),
                                  length,
                                  nbatch,
                                  istride,
                                  idist,
                                  ostride,
                                  odist,
                                  ioffset,
                                  ooffset);
                break;
            case fft_precision_double:
                copy_buffers_1to1(reinterpret_cast<const rocfft_complex<double>*>(input[0].data()),
                                  reinterpret_cast<rocfft_complex<double>*>(output[0].data()),
                                  length,
                                  nbatch,
                                  istride,
                                  idist,
                                  ostride,
                                  odist,
                                  ioffset,
                                  ooffset);
                break;
            }
            break;
        case fft_array_type_real:
        case fft_array_type_complex_planar:
        case fft_array_type_hermitian_planar:
            for(unsigned int idx = 0; idx < input.size(); ++idx)
            {
                switch(precision)
                {
                case fft_precision_half:
                    copy_buffers_1to1(reinterpret_cast<const rocfft_fp16*>(input[idx].data()),
                                      reinterpret_cast<rocfft_fp16*>(output[idx].data()),
                                      length,
                                      nbatch,
                                      istride,
                                      idist,
                                      ostride,
                                      odist,
                                      ioffset,
                                      ooffset);
                    break;
                case fft_precision_single:
                    copy_buffers_1to1(reinterpret_cast<const float*>(input[idx].data()),
                                      reinterpret_cast<float*>(output[idx].data()),
                                      length,
                                      nbatch,
                                      istride,
                                      idist,
                                      ostride,
                                      odist,
                                      ioffset,
                                      ooffset);
                    break;
                case fft_precision_double:
                    copy_buffers_1to1(reinterpret_cast<const double*>(input[idx].data()),
                                      reinterpret_cast<double*>(output[idx].data()),
                                      length,
                                      nbatch,
                                      istride,
                                      idist,
                                      ostride,
                                      odist,
                                      ioffset,
                                      ooffset);
                    break;
                }
            }
            break;
        default:
            throw std::runtime_error("Invalid data type");
        }
    }
    else if((itype == fft_array_type_complex_interleaved && otype == fft_array_type_complex_planar)
            || (itype == fft_array_type_hermitian_interleaved
                && otype == fft_array_type_hermitian_planar))
    {
        // copy 1to2
        switch(precision)
        {
        case fft_precision_half:
            copy_buffers_1to2(reinterpret_cast<const rocfft_complex<rocfft_fp16>*>(input[0].data()),
                              reinterpret_cast<rocfft_fp16*>(output[0].data()),
                              reinterpret_cast<rocfft_fp16*>(output[1].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              ostride,
                              odist,
                              ioffset,
                              ooffset);
            break;
        case fft_precision_single:
            copy_buffers_1to2(reinterpret_cast<const rocfft_complex<float>*>(input[0].data()),
                              reinterpret_cast<float*>(output[0].data()),
                              reinterpret_cast<float*>(output[1].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              ostride,
                              odist,
                              ioffset,
                              ooffset);
            break;
        case fft_precision_double:
            copy_buffers_1to2(reinterpret_cast<const rocfft_complex<double>*>(input[0].data()),
                              reinterpret_cast<double*>(output[0].data()),
                              reinterpret_cast<double*>(output[1].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              ostride,
                              odist,
                              ioffset,
                              ooffset);
            break;
        }
    }
    else if((itype == fft_array_type_complex_planar && otype == fft_array_type_complex_interleaved)
            || (itype == fft_array_type_hermitian_planar
                && otype == fft_array_type_hermitian_interleaved))
    {
        // copy 2 to 1
        switch(precision)
        {
        case fft_precision_half:
            copy_buffers_2to1(reinterpret_cast<const rocfft_fp16*>(input[0].data()),
                              reinterpret_cast<const rocfft_fp16*>(input[1].data()),
                              reinterpret_cast<rocfft_complex<rocfft_fp16>*>(output[0].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              ostride,
                              odist,
                              ioffset,
                              ooffset);
            break;
        case fft_precision_single:
            copy_buffers_2to1(reinterpret_cast<const float*>(input[0].data()),
                              reinterpret_cast<const float*>(input[1].data()),
                              reinterpret_cast<rocfft_complex<float>*>(output[0].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              ostride,
                              odist,
                              ioffset,
                              ooffset);
            break;
        case fft_precision_double:
            copy_buffers_2to1(reinterpret_cast<const double*>(input[0].data()),
                              reinterpret_cast<const double*>(input[1].data()),
                              reinterpret_cast<rocfft_complex<double>*>(output[0].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              ostride,
                              odist,
                              ioffset,
                              ooffset);
            break;
        }
    }
    else
    {
        throw std::runtime_error("Invalid input and output types.");
    }
}

// unroll arbitrary-dimension copy_buffers into specializations for 1-, 2-, 3-dimensions
template <typename Tint1, typename Tint2, typename Tint3>
inline void copy_buffers(const std::vector<hostbuf>& input,
                         std::vector<hostbuf>&       output,
                         const std::vector<Tint1>&   length,
                         const size_t                nbatch,
                         const fft_precision         precision,
                         const fft_array_type        itype,
                         const std::vector<Tint2>&   istride,
                         const size_t                idist,
                         const fft_array_type        otype,
                         const std::vector<Tint3>&   ostride,
                         const size_t                odist,
                         const std::vector<size_t>&  ioffset,
                         const std::vector<size_t>&  ooffset)
{
    switch(length.size())
    {
    case 1:
        return copy_buffers(input,
                            output,
                            length[0],
                            nbatch,
                            precision,
                            itype,
                            istride[0],
                            idist,
                            otype,
                            ostride[0],
                            odist,
                            ioffset,
                            ooffset);
    case 2:
        return copy_buffers(input,
                            output,
                            std::make_tuple(length[0], length[1]),
                            nbatch,
                            precision,
                            itype,
                            std::make_tuple(istride[0], istride[1]),
                            idist,
                            otype,
                            std::make_tuple(ostride[0], ostride[1]),
                            odist,
                            ioffset,
                            ooffset);
    case 3:
        return copy_buffers(input,
                            output,
                            std::make_tuple(length[0], length[1], length[2]),
                            nbatch,
                            precision,
                            itype,
                            std::make_tuple(istride[0], istride[1], istride[2]),
                            idist,
                            otype,
                            std::make_tuple(ostride[0], ostride[1], ostride[2]),
                            odist,
                            ioffset,
                            ooffset);
    case 4:
    {
        // treat 4D brick as 3D + batch, but this needs to be
        // unbatched until we add 4D support for copy_buffers
        if(nbatch != 1)
            throw std::runtime_error("cannot copy batched 4D bricks");
        return copy_buffers(input,
                            output,
                            std::make_tuple(length[1], length[2], length[3]),
                            length[0],
                            precision,
                            itype,
                            std::make_tuple(istride[1], istride[2], istride[3]),
                            istride[0],
                            otype,
                            std::make_tuple(ostride[1], ostride[2], istride[3]),
                            ostride[0],
                            ioffset,
                            ooffset);
    }
    default:
        abort();
    }
}

// Compute the L-infinity and L-2 distance between two buffers with strides istride and
// length idist between batches to a buffer with strides ostride and length odist between
// batches.  Both buffers are of complex type.

struct VectorNorms
{
    double l_2 = 0.0, l_inf = 0.0;
};

template <typename Tcomplex, typename Tint1, typename Tint2, typename Tint3>
inline VectorNorms distance_1to1_complex(const Tcomplex*                         input,
                                         const Tcomplex*                         output,
                                         const Tint1&                            whole_length,
                                         const size_t                            nbatch,
                                         const Tint2&                            istride,
                                         const size_t                            idist,
                                         const Tint3&                            ostride,
                                         const size_t                            odist,
                                         std::vector<std::pair<size_t, size_t>>* linf_failures,
                                         const double                            linf_cutoff,
                                         const std::vector<size_t>&              ioffset,
                                         const std::vector<size_t>&              ooffset,
                                         const double output_scalar = 1.0)
{
    double linf = 0.0;
    double l2   = 0.0;

    std::mutex                             linf_failure_lock;
    std::vector<std::pair<size_t, size_t>> linf_failures_private;

    const bool idx_equals_odx = istride == ostride && idist == odist;
    size_t     idx_base       = 0;
    size_t     odx_base       = 0;
    auto       partitions     = partition_colmajor(whole_length);
    for(size_t b = 0; b < nbatch; b++, idx_base += idist, odx_base += odist)
    {
#ifdef _OPENMP
#pragma omp parallel for reduction(max : linf) reduction(+ : l2) \
    num_threads(partitions.size()) private(linf_failures_private)
#endif
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            double     cur_linf = 0.0;
            double     cur_l2   = 0.0;
            auto       index    = partitions[part].first;
            const auto length   = partitions[part].second;

            do
            {
                const auto   idx = compute_index(index, istride, idx_base);
                const auto   odx = idx_equals_odx ? idx : compute_index(index, ostride, odx_base);
                const double rdiff
                    = std::abs(static_cast<double>(output[odx + ooffset[0]].real()) * output_scalar
                               - static_cast<double>(input[idx + ioffset[0]].real()));
                cur_linf = std::max(rdiff, cur_linf);
                if(cur_linf > linf_cutoff)
                {
                    std::pair<size_t, size_t> fval(b, idx);
                    if(linf_failures)
                        linf_failures_private.push_back(fval);
                }
                cur_l2 += rdiff * rdiff;

                const double idiff
                    = std::abs(static_cast<double>(output[odx + ooffset[0]].imag()) * output_scalar
                               - static_cast<double>(input[idx + ioffset[0]].imag()));
                cur_linf = std::max(idiff, cur_linf);
                if(cur_linf > linf_cutoff)
                {
                    std::pair<size_t, size_t> fval(b, idx);
                    if(linf_failures)
                        linf_failures_private.push_back(fval);
                }
                cur_l2 += idiff * idiff;

            } while(increment_rowmajor(index, length));
            linf = std::max(linf, cur_linf);
            l2 += cur_l2;

            if(linf_failures)
            {
                linf_failure_lock.lock();
                std::copy(linf_failures_private.begin(),
                          linf_failures_private.end(),
                          std::back_inserter(*linf_failures));
                linf_failure_lock.unlock();
            }
        }
    }
    return {.l_2 = sqrt(l2), .l_inf = linf};
}

// Compute the L-infinity and L-2 distance between two buffers with strides istride and
// length idist between batches to a buffer with strides ostride and length odist between
// batches.  Both buffers are of real type.
template <typename Tfloat, typename Tint1, typename Tint2, typename Tint3>
inline VectorNorms distance_1to1_real(const Tfloat*                           input,
                                      const Tfloat*                           output,
                                      const Tint1&                            whole_length,
                                      const size_t                            nbatch,
                                      const Tint2&                            istride,
                                      const size_t                            idist,
                                      const Tint3&                            ostride,
                                      const size_t                            odist,
                                      std::vector<std::pair<size_t, size_t>>* linf_failures,
                                      const double                            linf_cutoff,
                                      const std::vector<size_t>&              ioffset,
                                      const std::vector<size_t>&              ooffset,
                                      const double                            output_scalar = 1.0)
{
    double linf = 0.0;
    double l2   = 0.0;

    std::mutex                             linf_failure_lock;
    std::vector<std::pair<size_t, size_t>> linf_failures_private;

    const bool idx_equals_odx = istride == ostride && idist == odist;
    size_t     idx_base       = 0;
    size_t     odx_base       = 0;
    auto       partitions     = partition_rowmajor(whole_length);
    for(size_t b = 0; b < nbatch; b++, idx_base += idist, odx_base += odist)
    {
#ifdef _OPENMP
#pragma omp parallel for reduction(max : linf) reduction(+ : l2) \
    num_threads(partitions.size()) private(linf_failures_private)
#endif
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            double     cur_linf = 0.0;
            double     cur_l2   = 0.0;
            auto       index    = partitions[part].first;
            const auto length   = partitions[part].second;
            do
            {
                const auto   idx = compute_index(index, istride, idx_base);
                const auto   odx = idx_equals_odx ? idx : compute_index(index, ostride, odx_base);
                const double diff
                    = std::abs(static_cast<double>(output[odx + ooffset[0]]) * output_scalar
                               - static_cast<double>(input[idx + ioffset[0]]));
                cur_linf = std::max(diff, cur_linf);
                if(cur_linf > linf_cutoff)
                {
                    std::pair<size_t, size_t> fval(b, idx);
                    if(linf_failures)
                        linf_failures_private.push_back(fval);
                }
                cur_l2 += diff * diff;

            } while(increment_rowmajor(index, length));
            linf = std::max(linf, cur_linf);
            l2 += cur_l2;

            if(linf_failures)
            {
                linf_failure_lock.lock();
                std::copy(linf_failures_private.begin(),
                          linf_failures_private.end(),
                          std::back_inserter(*linf_failures));
                linf_failure_lock.unlock();
            }
        }
    }
    return {.l_2 = sqrt(l2), .l_inf = linf};
}

// Compute the L-infinity and L-2 distance between two buffers with strides istride and
// length idist between batches to a buffer with strides ostride and length odist between
// batches.  input is complex-interleaved, output is complex-planar.
template <typename Tval, typename Tint1, typename T2, typename T3>
inline VectorNorms distance_1to2(const rocfft_complex<Tval>*             input,
                                 const Tval*                             output0,
                                 const Tval*                             output1,
                                 const Tint1&                            whole_length,
                                 const size_t                            nbatch,
                                 const T2&                               istride,
                                 const size_t                            idist,
                                 const T3&                               ostride,
                                 const size_t                            odist,
                                 std::vector<std::pair<size_t, size_t>>* linf_failures,
                                 const double                            linf_cutoff,
                                 const std::vector<size_t>&              ioffset,
                                 const std::vector<size_t>&              ooffset,
                                 const double                            output_scalar = 1.0)
{
    double linf = 0.0;
    double l2   = 0.0;

    std::mutex                             linf_failure_lock;
    std::vector<std::pair<size_t, size_t>> linf_failures_private;

    const bool idx_equals_odx = istride == ostride && idist == odist;
    size_t     idx_base       = 0;
    size_t     odx_base       = 0;
    auto       partitions     = partition_rowmajor(whole_length);
    for(size_t b = 0; b < nbatch; b++, idx_base += idist, odx_base += odist)
    {
#ifdef _OPENMP
#pragma omp parallel for reduction(max : linf) reduction(+ : l2) \
    num_threads(partitions.size()) private(linf_failures_private)
#endif
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            double     cur_linf = 0.0;
            double     cur_l2   = 0.0;
            auto       index    = partitions[part].first;
            const auto length   = partitions[part].second;
            do
            {
                const auto   idx = compute_index(index, istride, idx_base);
                const auto   odx = idx_equals_odx ? idx : compute_index(index, ostride, odx_base);
                const double rdiff
                    = std::abs(static_cast<double>(output0[odx + ooffset[0]]) * output_scalar
                               - static_cast<double>(input[idx + ioffset[0]].real()));
                cur_linf = std::max(rdiff, cur_linf);
                if(cur_linf > linf_cutoff)
                {
                    std::pair<size_t, size_t> fval(b, idx);
                    if(linf_failures)
                        linf_failures_private.push_back(fval);
                }
                cur_l2 += rdiff * rdiff;

                const double idiff
                    = std::abs(static_cast<double>(output1[odx + ooffset[1]]) * output_scalar
                               - static_cast<double>(input[idx + ioffset[0]].imag()));
                cur_linf = std::max(idiff, cur_linf);
                if(cur_linf > linf_cutoff)
                {
                    std::pair<size_t, size_t> fval(b, idx);
                    if(linf_failures)
                        linf_failures_private.push_back(fval);
                }
                cur_l2 += idiff * idiff;

            } while(increment_rowmajor(index, length));
            linf = std::max(linf, cur_linf);
            l2 += cur_l2;

            if(linf_failures)
            {
                linf_failure_lock.lock();
                std::copy(linf_failures_private.begin(),
                          linf_failures_private.end(),
                          std::back_inserter(*linf_failures));
                linf_failure_lock.unlock();
            }
        }
    }
    return {.l_2 = sqrt(l2), .l_inf = linf};
}

// Compute the L-inifnity and L-2 distance between two buffers of dimension length and
// with types given by itype, otype, and precision.
template <typename Tint1, typename Tint2, typename Tint3>
inline VectorNorms distance(const std::vector<hostbuf>&             input,
                            const std::vector<hostbuf>&             output,
                            const Tint1&                            length,
                            const size_t                            nbatch,
                            const fft_precision                     precision,
                            const fft_array_type                    itype,
                            const Tint2&                            istride,
                            const size_t                            idist,
                            const fft_array_type                    otype,
                            const Tint3&                            ostride,
                            const size_t                            odist,
                            std::vector<std::pair<size_t, size_t>>* linf_failures,
                            const double                            linf_cutoff,
                            const std::vector<size_t>&              ioffset,
                            const std::vector<size_t>&              ooffset,
                            const double                            output_scalar = 1.0)
{
    VectorNorms dist;

    if(itype == otype)
    {
        switch(itype)
        {
        case fft_array_type_complex_interleaved:
        case fft_array_type_hermitian_interleaved:
            switch(precision)
            {
            case fft_precision_half:
                dist = distance_1to1_complex(
                    reinterpret_cast<const rocfft_complex<rocfft_fp16>*>(input[0].data()),
                    reinterpret_cast<const rocfft_complex<rocfft_fp16>*>(output[0].data()),
                    length,
                    nbatch,
                    istride,
                    idist,
                    ostride,
                    odist,
                    linf_failures,
                    linf_cutoff,
                    ioffset,
                    ooffset,
                    output_scalar);
                break;
            case fft_precision_single:
                dist = distance_1to1_complex(
                    reinterpret_cast<const rocfft_complex<float>*>(input[0].data()),
                    reinterpret_cast<const rocfft_complex<float>*>(output[0].data()),
                    length,
                    nbatch,
                    istride,
                    idist,
                    ostride,
                    odist,
                    linf_failures,
                    linf_cutoff,
                    ioffset,
                    ooffset,
                    output_scalar);
                break;
            case fft_precision_double:
                dist = distance_1to1_complex(
                    reinterpret_cast<const rocfft_complex<double>*>(input[0].data()),
                    reinterpret_cast<const rocfft_complex<double>*>(output[0].data()),
                    length,
                    nbatch,
                    istride,
                    idist,
                    ostride,
                    odist,
                    linf_failures,
                    linf_cutoff,
                    ioffset,
                    ooffset,
                    output_scalar);
                break;
            }
            dist.l_2 *= dist.l_2;
            break;
        case fft_array_type_real:
        case fft_array_type_complex_planar:
        case fft_array_type_hermitian_planar:
            for(unsigned int idx = 0; idx < input.size(); ++idx)
            {
                VectorNorms d;
                switch(precision)
                {
                case fft_precision_half:
                    d = distance_1to1_real(reinterpret_cast<const rocfft_fp16*>(input[idx].data()),
                                           reinterpret_cast<const rocfft_fp16*>(output[idx].data()),
                                           length,
                                           nbatch,
                                           istride,
                                           idist,
                                           ostride,
                                           odist,
                                           linf_failures,
                                           linf_cutoff,
                                           ioffset,
                                           ooffset,
                                           output_scalar);
                    break;
                case fft_precision_single:
                    d = distance_1to1_real(reinterpret_cast<const float*>(input[idx].data()),
                                           reinterpret_cast<const float*>(output[idx].data()),
                                           length,
                                           nbatch,
                                           istride,
                                           idist,
                                           ostride,
                                           odist,
                                           linf_failures,
                                           linf_cutoff,
                                           ioffset,
                                           ooffset,
                                           output_scalar);
                    break;
                case fft_precision_double:
                    d = distance_1to1_real(reinterpret_cast<const double*>(input[idx].data()),
                                           reinterpret_cast<const double*>(output[idx].data()),
                                           length,
                                           nbatch,
                                           istride,
                                           idist,
                                           ostride,
                                           odist,
                                           linf_failures,
                                           linf_cutoff,
                                           ioffset,
                                           ooffset,
                                           output_scalar);
                    break;
                }
                dist.l_inf = std::max(d.l_inf, dist.l_inf);
                dist.l_2 += d.l_2 * d.l_2;
            }
            break;
        default:
            throw std::runtime_error("Invalid input and output types.");
        }
    }
    else if((itype == fft_array_type_complex_interleaved && otype == fft_array_type_complex_planar)
            || (itype == fft_array_type_hermitian_interleaved
                && otype == fft_array_type_hermitian_planar))
    {
        switch(precision)
        {
        case fft_precision_half:
            dist = distance_1to2(
                reinterpret_cast<const rocfft_complex<rocfft_fp16>*>(input[0].data()),
                reinterpret_cast<const rocfft_fp16*>(output[0].data()),
                reinterpret_cast<const rocfft_fp16*>(output[1].data()),
                length,
                nbatch,
                istride,
                idist,
                ostride,
                odist,
                linf_failures,
                linf_cutoff,
                ioffset,
                ooffset,
                output_scalar);
            break;
        case fft_precision_single:
            dist = distance_1to2(reinterpret_cast<const rocfft_complex<float>*>(input[0].data()),
                                 reinterpret_cast<const float*>(output[0].data()),
                                 reinterpret_cast<const float*>(output[1].data()),
                                 length,
                                 nbatch,
                                 istride,
                                 idist,
                                 ostride,
                                 odist,
                                 linf_failures,
                                 linf_cutoff,
                                 ioffset,
                                 ooffset,
                                 output_scalar);
            break;
        case fft_precision_double:
            dist = distance_1to2(reinterpret_cast<const rocfft_complex<double>*>(input[0].data()),
                                 reinterpret_cast<const double*>(output[0].data()),
                                 reinterpret_cast<const double*>(output[1].data()),
                                 length,
                                 nbatch,
                                 istride,
                                 idist,
                                 ostride,
                                 odist,
                                 linf_failures,
                                 linf_cutoff,
                                 ioffset,
                                 ooffset,
                                 output_scalar);
            break;
        }
        dist.l_2 *= dist.l_2;
    }
    else if((itype == fft_array_type_complex_planar && otype == fft_array_type_complex_interleaved)
            || (itype == fft_array_type_hermitian_planar
                && otype == fft_array_type_hermitian_interleaved))
    {
        switch(precision)
        {
        case fft_precision_half:
            dist = distance_1to2(
                reinterpret_cast<const rocfft_complex<rocfft_fp16>*>(output[0].data()),
                reinterpret_cast<const rocfft_fp16*>(input[0].data()),
                reinterpret_cast<const rocfft_fp16*>(input[1].data()),
                length,
                nbatch,
                ostride,
                odist,
                istride,
                idist,
                linf_failures,
                linf_cutoff,
                ioffset,
                ooffset,
                output_scalar);
            break;
        case fft_precision_single:
            dist = distance_1to2(reinterpret_cast<const rocfft_complex<float>*>(output[0].data()),
                                 reinterpret_cast<const float*>(input[0].data()),
                                 reinterpret_cast<const float*>(input[1].data()),
                                 length,
                                 nbatch,
                                 ostride,
                                 odist,
                                 istride,
                                 idist,
                                 linf_failures,
                                 linf_cutoff,
                                 ioffset,
                                 ooffset,
                                 output_scalar);
            break;
        case fft_precision_double:
            dist = distance_1to2(reinterpret_cast<const rocfft_complex<double>*>(output[0].data()),
                                 reinterpret_cast<const double*>(input[0].data()),
                                 reinterpret_cast<const double*>(input[1].data()),
                                 length,
                                 nbatch,
                                 ostride,
                                 odist,
                                 istride,
                                 idist,
                                 linf_failures,
                                 linf_cutoff,
                                 ioffset,
                                 ooffset,
                                 output_scalar);
            break;
        }
        dist.l_2 *= dist.l_2;
    }
    else
    {
        throw std::runtime_error("Invalid input and output types.");
    }
    dist.l_2 = sqrt(dist.l_2);
    return dist;
}

// check if the specified length + stride/dist is contiguous
template <typename Tint1, typename Tint2>
bool is_contiguous_rowmajor(const std::vector<Tint1>& length,
                            const std::vector<Tint2>& stride,
                            size_t                    dist)
{
    size_t expected_stride = 1;
    auto   stride_it       = stride.rbegin();
    auto   length_it       = length.rbegin();
    for(; stride_it != stride.rend() && length_it != length.rend(); ++stride_it, ++length_it)
    {
        if(*stride_it != expected_stride)
            return false;
        expected_stride *= *length_it;
    }
    return expected_stride == dist;
}

// Unroll arbitrary-dimension distance into specializations for 1-, 2-, 3-dimensions
template <typename Tint1, typename Tint2, typename Tint3>
inline VectorNorms distance(const std::vector<hostbuf>&             input,
                            const std::vector<hostbuf>&             output,
                            std::vector<Tint1>                      length,
                            size_t                                  nbatch,
                            const fft_precision                     precision,
                            const fft_array_type                    itype,
                            std::vector<Tint2>                      istride,
                            const size_t                            idist,
                            const fft_array_type                    otype,
                            std::vector<Tint3>                      ostride,
                            const size_t                            odist,
                            std::vector<std::pair<size_t, size_t>>* linf_failures,
                            const double                            linf_cutoff,
                            const std::vector<size_t>&              ioffset,
                            const std::vector<size_t>&              ooffset,
                            const double                            output_scalar = 1.0)
{
    // If istride and ostride are both contiguous, collapse them down
    // to one dimension.  Index calculation is simpler (and faster)
    // in the 1D case.
    if(is_contiguous_rowmajor(length, istride, idist)
       && is_contiguous_rowmajor(length, ostride, odist))
    {
        length  = {product(length.begin(), length.end()) * nbatch};
        istride = {static_cast<Tint2>(1)};
        ostride = {static_cast<Tint3>(1)};
        nbatch  = 1;
    }

    switch(length.size())
    {
    case 1:
        return distance(input,
                        output,
                        length[0],
                        nbatch,
                        precision,
                        itype,
                        istride[0],
                        idist,
                        otype,
                        ostride[0],
                        odist,
                        linf_failures,
                        linf_cutoff,
                        ioffset,
                        ooffset,
                        output_scalar);
    case 2:
        return distance(input,
                        output,
                        std::make_tuple(length[0], length[1]),
                        nbatch,
                        precision,
                        itype,
                        std::make_tuple(istride[0], istride[1]),
                        idist,
                        otype,
                        std::make_tuple(ostride[0], ostride[1]),
                        odist,
                        linf_failures,
                        linf_cutoff,
                        ioffset,
                        ooffset,
                        output_scalar);
    case 3:
        return distance(input,
                        output,
                        std::make_tuple(length[0], length[1], length[2]),
                        nbatch,
                        precision,
                        itype,
                        std::make_tuple(istride[0], istride[1], istride[2]),
                        idist,
                        otype,
                        std::make_tuple(ostride[0], ostride[1], ostride[2]),
                        odist,
                        linf_failures,
                        linf_cutoff,
                        ioffset,
                        ooffset,
                        output_scalar);
    default:
        abort();
    }
}

// Compute the L-infinity and L-2 norm of a buffer with strides istride and
// length idist.  Data is rocfft_complex.
template <typename Tcomplex, typename T1, typename T2>
inline VectorNorms norm_complex(const Tcomplex*            input,
                                const T1&                  whole_length,
                                const size_t               nbatch,
                                const T2&                  istride,
                                const size_t               idist,
                                const std::vector<size_t>& offset)
{
    double linf = 0.0;
    double l2   = 0.0;

    size_t idx_base   = 0;
    auto   partitions = partition_rowmajor(whole_length);
    for(size_t b = 0; b < nbatch; b++, idx_base += idist)
    {
#ifdef _OPENMP
#pragma omp parallel for reduction(max : linf) reduction(+ : l2) num_threads(partitions.size())
#endif
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            double     cur_linf = 0.0;
            double     cur_l2   = 0.0;
            auto       index    = partitions[part].first;
            const auto length   = partitions[part].second;
            do
            {
                const auto idx = compute_index(index, istride, idx_base);

                const double rval = std::abs(static_cast<double>(input[idx + offset[0]].real()));
                cur_linf          = std::max(rval, cur_linf);
                cur_l2 += rval * rval;

                const double ival = std::abs(static_cast<double>(input[idx + offset[0]].imag()));
                cur_linf          = std::max(ival, cur_linf);
                cur_l2 += ival * ival;

            } while(increment_rowmajor(index, length));
            linf = std::max(linf, cur_linf);
            l2 += cur_l2;
        }
    }
    return {.l_2 = sqrt(l2), .l_inf = linf};
}

// Compute the L-infinity and L-2 norm of abuffer with strides istride and
// length idist.  Data is real-valued.
template <typename Tfloat, typename T1, typename T2>
inline VectorNorms norm_real(const Tfloat*              input,
                             const T1&                  whole_length,
                             const size_t               nbatch,
                             const T2&                  istride,
                             const size_t               idist,
                             const std::vector<size_t>& offset)
{
    double linf = 0.0;
    double l2   = 0.0;

    size_t idx_base   = 0;
    auto   partitions = partition_rowmajor(whole_length);
    for(size_t b = 0; b < nbatch; b++, idx_base += idist)
    {
#ifdef _OPENMP
#pragma omp parallel for reduction(max : linf) reduction(+ : l2) num_threads(partitions.size())
#endif
        for(size_t part = 0; part < partitions.size(); ++part)
        {
            double     cur_linf = 0.0;
            double     cur_l2   = 0.0;
            auto       index    = partitions[part].first;
            const auto length   = partitions[part].second;
            do
            {
                const auto   idx = compute_index(index, istride, idx_base);
                const double val = std::abs(static_cast<double>(input[idx + offset[0]]));
                cur_linf         = std::max(val, cur_linf);
                cur_l2 += val * val;

            } while(increment_rowmajor(index, length));
            linf = std::max(linf, cur_linf);
            l2 += cur_l2;
        }
    }
    return {.l_2 = sqrt(l2), .l_inf = linf};
}

// Compute the L-infinity and L-2 norm of abuffer with strides istride and
// length idist.  Data format is given by precision and itype.
template <typename T1, typename T2>
inline VectorNorms norm(const std::vector<hostbuf>& input,
                        const T1&                   length,
                        const size_t                nbatch,
                        const fft_precision         precision,
                        const fft_array_type        itype,
                        const T2&                   istride,
                        const size_t                idist,
                        const std::vector<size_t>&  offset)
{
    VectorNorms norm;

    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
        switch(precision)
        {
        case fft_precision_half:
            norm = norm_complex(
                reinterpret_cast<const rocfft_complex<rocfft_fp16>*>(input[0].data()),
                length,
                nbatch,
                istride,
                idist,
                offset);
            break;
        case fft_precision_single:
            norm = norm_complex(reinterpret_cast<const rocfft_complex<float>*>(input[0].data()),
                                length,
                                nbatch,
                                istride,
                                idist,
                                offset);
            break;
        case fft_precision_double:
            norm = norm_complex(reinterpret_cast<const rocfft_complex<double>*>(input[0].data()),
                                length,
                                nbatch,
                                istride,
                                idist,
                                offset);
            break;
        }
        norm.l_2 *= norm.l_2;
        break;
    case fft_array_type_real:
    case fft_array_type_complex_planar:
    case fft_array_type_hermitian_planar:
        for(unsigned int idx = 0; idx < input.size(); ++idx)
        {
            VectorNorms n;
            switch(precision)
            {
            case fft_precision_half:
                n = norm_real(reinterpret_cast<const rocfft_fp16*>(input[idx].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              offset);
                break;
            case fft_precision_single:
                n = norm_real(reinterpret_cast<const float*>(input[idx].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              offset);
                break;
            case fft_precision_double:
                n = norm_real(reinterpret_cast<const double*>(input[idx].data()),
                              length,
                              nbatch,
                              istride,
                              idist,
                              offset);
                break;
            }
            norm.l_inf = std::max(n.l_inf, norm.l_inf);
            norm.l_2 += n.l_2 * n.l_2;
        }
        break;
    default:
        throw std::runtime_error("Invalid data type");
    }

    norm.l_2 = sqrt(norm.l_2);
    return norm;
}

// Unroll arbitrary-dimension norm into specializations for 1-, 2-, 3-dimensions
template <typename T1, typename T2>
inline VectorNorms norm(const std::vector<hostbuf>& input,
                        std::vector<T1>             length,
                        size_t                      nbatch,
                        const fft_precision         precision,
                        const fft_array_type        type,
                        std::vector<T2>             stride,
                        const size_t                dist,
                        const std::vector<size_t>&  offset)
{
    // If stride is contiguous, collapse it down to one dimension.
    // Index calculation is simpler (and faster) in the 1D case.
    if(is_contiguous_rowmajor(length, stride, dist))
    {
        length = {product(length.begin(), length.end()) * nbatch};
        stride = {static_cast<T2>(1)};
        nbatch = 1;
    }

    switch(length.size())
    {
    case 1:
        return norm(input, length[0], nbatch, precision, type, stride[0], dist, offset);
    case 2:
        return norm(input,
                    std::make_tuple(length[0], length[1]),
                    nbatch,
                    precision,
                    type,
                    std::make_tuple(stride[0], stride[1]),
                    dist,
                    offset);
    case 3:
        return norm(input,
                    std::make_tuple(length[0], length[1], length[2]),
                    nbatch,
                    precision,
                    type,
                    std::make_tuple(stride[0], stride[1], stride[2]),
                    dist,
                    offset);
    default:
        abort();
    }
}

// returns byte_sizes.size() host buffers of respective sizes byte_sizes[0], byte_sizes[1], ...
static std::vector<hostbuf> allocate_host_buffer(const std::vector<size_t>& byte_sizes)
{
    std::vector<hostbuf> buffers(byte_sizes.size());
    for(unsigned int i = 0; i < byte_sizes.size(); ++i)
    {
        buffers[i].alloc(byte_sizes[i]);
    }
    return buffers;
}

static std::vector<hostbuf> allocate_host_buffer(const fft_precision        precision,
                                                 const fft_array_type       type,
                                                 const std::vector<size_t>& size)
{
    std::vector<size_t> byte_sizes = size;
    for(auto& sz : byte_sizes)
        sz *= var_size<size_t>(precision, type);
    return allocate_host_buffer(byte_sizes);
}

// Check if the required buffers fit in the device vram.
inline bool vram_fits_problem(const std::vector<size_t>& prob_size,
                              const std::vector<size_t>& vram_avail)
{
    if(prob_size.size() != vram_avail.size())
        throw std::runtime_error("prob/vram size mismatch");

    // We keep a small margin of error for fitting the problem into vram:
    const size_t extra = 1 << 27;

    for(size_t i = 0; i < prob_size.size(); ++i)
    {
        if(prob_size[i] + extra > vram_avail[i])
            return false;
    }
    return true;
}

// set input for a brick in a field
// functor to search for bricks on a rank, in a container of bricks
// sorted by rank
struct match_rank
{
    bool operator()(const fft_params::fft_brick& b, int rank) const
    {
        return b.rank < rank;
    }
    bool operator()(int rank, const fft_params::fft_brick& b) const
    {
        return rank < b.rank;
    }
};

// Initialize input for the bricks on the specified comm rank
// (assumed to be the local rank)
template <typename Tparams, typename Tbuff>
void init_local_input(int                                       comm_rank,
                      const Tparams&                            params,
                      const std::vector<fft_params::fft_brick>& bricks,
                      size_t                                    elem_size,
                      const std::vector<void*>&                 input_ptrs)
{
    // get bricks for this rank
    auto range = std::equal_range(bricks.begin(), bricks.end(), comm_rank, match_rank());

    const bool is_planar = params.itype == fft_array_type_complex_planar
                           || params.itype == fft_array_type_hermitian_planar;

    size_t ptr_idx = 0;
    for(auto brick = range.first; brick != range.second; ++brick, ++ptr_idx)
    {
        rocfft_scoped_device dev(brick->device);

        // some utility code below needs batch separated from brick lengths
        std::vector<size_t> brick_len_nobatch = brick->length();
        auto                brick_batch       = brick_len_nobatch.front();
        brick_len_nobatch.erase(brick_len_nobatch.begin());
        std::vector<size_t> brick_stride_nobatch = brick->stride;
        auto                brick_dist           = brick_stride_nobatch.front();
        brick_stride_nobatch.erase(brick_stride_nobatch.begin());
        std::vector<size_t> brick_lower_nobatch = brick->lower;
        auto                brick_lower_batch   = brick_lower_nobatch.front();
        brick_lower_nobatch.erase(brick_lower_nobatch.begin());

        auto contiguous_stride = params.compute_stride(params.ilength());
        auto contiguous_dist   = params.compute_idist();

        std::vector<Tbuff> bufvec(1);
        size_t             brick_size_bytes
            = compute_ptrdiff(brick->length(), brick->stride) * elem_size / (is_planar ? 2 : 1);
        bufvec.back() = Tbuff::make_nonowned(input_ptrs[ptr_idx], brick_size_bytes);
        // grab a second pointer for planar
        if(is_planar)
        {
            ++ptr_idx;
            bufvec.push_back(Tbuff::make_nonowned(input_ptrs[ptr_idx], brick_size_bytes));
        }

        // generate data (in device mem)
        switch(params.precision)
        {
        case fft_precision_half:
            set_input<Tbuff, rocfft_fp16>(bufvec,
                                          params.igen,
                                          params.itype,
                                          brick_len_nobatch,
                                          brick_len_nobatch,
                                          brick_stride_nobatch,
                                          params.ioffset,
                                          brick_dist,
                                          brick_batch,
                                          get_curr_device_prop(),
                                          brick_lower_nobatch,
                                          brick_lower_batch,
                                          contiguous_stride,
                                          contiguous_dist);
            break;
        case fft_precision_single:
            set_input<Tbuff, float>(bufvec,
                                    params.igen,
                                    params.itype,
                                    brick_len_nobatch,
                                    brick_len_nobatch,
                                    brick_stride_nobatch,
                                    params.ioffset,
                                    brick_dist,
                                    brick_batch,
                                    get_curr_device_prop(),
                                    brick_lower_nobatch,
                                    brick_lower_batch,
                                    contiguous_stride,
                                    contiguous_dist);
            break;
        case fft_precision_double:
            set_input<Tbuff, double>(bufvec,
                                     params.igen,
                                     params.itype,
                                     brick_len_nobatch,
                                     brick_len_nobatch,
                                     brick_stride_nobatch,
                                     params.ioffset,
                                     brick_dist,
                                     brick_batch,
                                     get_curr_device_prop(),
                                     brick_lower_nobatch,
                                     brick_lower_batch,
                                     contiguous_stride,
                                     contiguous_dist);
            break;
        }
    }
}

#endif
