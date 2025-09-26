// Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
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
#include "../shared/data_gen_device.h"
#include "../shared/data_gen_host.h"
#include "../shared/device_properties.h"
#include "../shared/gpubuf.h"
#include "../shared/printbuffer.h"
#include "../shared/ptrdiff.h"
#include "../shared/rocfft_complex.h"

enum fft_status
{
    fft_status_success,
    fft_status_failure,
    fft_status_invalid_arg_value,
    fft_status_invalid_dimensions,
    fft_status_invalid_array_type,
    fft_status_invalid_strides,
    fft_status_invalid_distance,
    fft_status_invalid_offset,
    fft_status_invalid_work_buffer,
};

enum fft_transform_type
{
    fft_transform_type_complex_forward,
    fft_transform_type_complex_inverse,
    fft_transform_type_real_forward,
    fft_transform_type_real_inverse,
};

enum fft_precision
{
    fft_precision_half,
    fft_precision_single,
    fft_precision_double,
};

enum fft_io
{
    fft_io_in,
    fft_io_out
};

static constexpr bool is_real(const fft_transform_type& dft_type)
{
    return dft_type == fft_transform_type_real_forward
           || dft_type == fft_transform_type_real_inverse;
}
static constexpr bool is_complex(const fft_transform_type& dft_type)
{
    return !is_real(dft_type);
}
static constexpr bool is_fwd(const fft_transform_type& dft_type)
{
    return dft_type == fft_transform_type_real_forward
           || dft_type == fft_transform_type_complex_forward;
}
static constexpr bool is_bwd(const fft_transform_type& dft_type)
{
    return !is_fwd(dft_type);
}

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

enum fft_auto_allocation
{
    fft_auto_allocation_on,
    fft_auto_allocation_off,
    fft_auto_allocation_default
};

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

// fft_input_generator: linearly spaced sequence in [-0.5,0.5]
// fft_input_random_generator: pseudo-random sequence in [-0.5,0.5]
enum fft_input_generator
{
    fft_input_random_generator_device,
    fft_input_random_generator_host,
    fft_input_generator_device,
    fft_input_generator_host,
};

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
    if(gen == fft_input_random_generator_device || gen == fft_input_generator_device)
        throw std::runtime_error(
            "Device input generation is not available, as hipRAND support is not enabled");
#endif
    return true;
}

enum fft_array_type
{
    fft_array_type_complex_interleaved,
    fft_array_type_complex_planar,
    fft_array_type_real,
    fft_array_type_hermitian_interleaved,
    fft_array_type_hermitian_planar,
    fft_array_type_unset,
};

enum fft_result_placement
{
    fft_placement_inplace,
    fft_placement_notinplace,
};

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

#ifdef USE_HIPRAND
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
    auto isize = count_iters(whole_length) * nbatch;

    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        auto ibuffer = (rocfft_complex<Tfloat>*)input[0].data();

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
        auto ibuffer_real = (Tfloat*)input[0].data();
        auto ibuffer_imag = (Tfloat*)input[1].data();

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

        if(itype == fft_array_type_hermitian_planar)
            impose_hermitian_symmetry_planar(
                length, ilength, istride, idist, nbatch, ibuffer_real, ibuffer_imag, deviceProp);

        break;
    }
    case fft_array_type_real:
    {
        auto ibuffer = (Tfloat*)input[0].data();

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

        break;
    }
    default:
        throw std::runtime_error("Input layout format not yet supported");
    }
}
#endif // USE_HIPRAND

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
    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        if(igen == fft_input_generator_host)
            generate_interleaved_data<Tfloat>(input, whole_length, whole_stride, idist, nbatch);
        else if(igen == fft_input_random_generator_host)
            generate_random_interleaved_data<Tfloat>(input,
                                                     whole_length,
                                                     whole_stride,
                                                     idist,
                                                     nbatch,
                                                     field_lower,
                                                     field_lower_batch,
                                                     field_contig_stride,
                                                     field_contig_dist);

        if(itype == fft_array_type_hermitian_interleaved)
            impose_hermitian_symmetry_interleaved<Tfloat>(input, length, istride, idist, nbatch);

        break;
    }
    case fft_array_type_complex_planar:
    case fft_array_type_hermitian_planar:
    {
        if(igen == fft_input_generator_host)
            generate_planar_data<Tfloat>(input, whole_length, whole_stride, idist, nbatch);
        else if(igen == fft_input_random_generator_host)
            generate_random_planar_data<Tfloat>(input,
                                                whole_length,
                                                whole_stride,
                                                idist,
                                                nbatch,
                                                field_lower,
                                                field_lower_batch,
                                                field_contig_stride,
                                                field_contig_dist);

        if(itype == fft_array_type_hermitian_planar)
            impose_hermitian_symmetry_planar<Tfloat>(input, length, istride, idist, nbatch);

        break;
    }
    case fft_array_type_real:
    {
        if(igen == fft_input_generator_host)
            generate_real_data<Tfloat>(input, whole_length, whole_stride, idist, nbatch);
        else if(igen == fft_input_random_generator_host)
            generate_random_real_data<Tfloat>(input,
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
        // accomadate the output, which is slightly larger.
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
                if(istride[i] != ostride[i])
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
                    std::cout << " ostride0:";
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

            if((transform_type == fft_transform_type_real_forward
                || transform_type == fft_transform_type_real_inverse)
               && (istride.back() != 1 || ostride.back() != 1))
            {
                // In-place real/complex transforms require unit strides.
                if(verbose)
                {
                    std::cout
                        << "istride.back(): " << istride.back()
                        << " ostride.back(): " << ostride.back()
                        << " must be unitary for in-place real/complex transforms: skipping test"
                        << std::endl;
                }
                return false;
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
        auto is_zero                = [](const decltype(ioffset)::value_type& i) { return i == 0; };
        bool default_layout_is_used = std::all_of(ioffset.begin(), ioffset.end(), is_zero)
                                      && std::all_of(ooffset.begin(), ooffset.end(), is_zero);
        size_t default_ival = 1;
        size_t default_oval = 1;
        default_layout_is_used
            &= (default_ival == istride.back()) && (default_oval == ostride.back());
        switch(transform_type)
        {
        case fft_transform_type_complex_forward:
        case fft_transform_type_complex_inverse:
        {
            default_ival *= length.back();
            default_oval *= length.back();
            break;
        }
        case fft_transform_type_real_forward:
        case fft_transform_type_real_inverse:
        {
            const size_t hermitian_symmetric_length = length.back() / 2 + 1;
            size_t*      ptr_default_real_val       = &default_ival;
            size_t*      ptr_default_cmplx_val      = &default_oval;
            if(transform_type == fft_transform_type_real_inverse)
                std::swap(ptr_default_real_val, ptr_default_cmplx_val);
            *ptr_default_cmplx_val *= hermitian_symmetric_length;
            if(placement == fft_placement_inplace)
                *ptr_default_real_val *= 2 * hermitian_symmetric_length; // padding to be used
            else
                *ptr_default_real_val *= length.back();
            break;
        }
        default:
            throw std::runtime_error("Invalid transform type");
        }
        // check strides for multi-dimensional cases
        for(int stride_idx = static_cast<int>(dim()) - 2; stride_idx >= 0 && default_layout_is_used;
            stride_idx--)
        {
            default_layout_is_used
                &= (default_ival == istride[stride_idx]) && (default_oval == ostride[stride_idx]);
            default_ival *= length[stride_idx];
            default_oval *= length[stride_idx];
        }
        // check distances
        default_layout_is_used &= (idist == default_ival) && (odist == default_oval);
        return default_layout_is_used;
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

    virtual fft_status set_callbacks(void*  load_cb_host,
                                     void*  load_cb_data,
                                     void*  store_cb_host,
                                     void*  store_cb_data,
                                     size_t load_cb_shared_mem_bytes,
                                     size_t store_cb_shared_mem_bytes)
    {
        return fft_status_success;
    }

    virtual fft_status execute(void** in, void** out)
    {
        return fft_status_success;
    };

    size_t fft_params_vram_footprint()
    {
        return fft_params::vram_footprint();
    }

    virtual size_t vram_footprint()
    {
        const auto ibuf_size = ibuffer_sizes();
        size_t val = std::accumulate(ibuf_size.begin(), ibuf_size.end(), static_cast<size_t>(1));
        if(placement == fft_placement_notinplace)
        {
            const auto obuf_size = obuffer_sizes();
            val += std::accumulate(obuf_size.begin(), obuf_size.end(), static_cast<size_t>(1));
        }
        return val;
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

    // prepare for multi-GPU transform.  cpu_input has contiguous
    // input on the host, ibuffer has device input, which can be
    // discarded if it's turned into multi-GPU bricks.  pibuffer,
    // pobuffer are the pointers that will be passed to the FFT
    // library's "execute" API.
    virtual void multi_gpu_prepare(std::vector<hostbuf>& cpu_input,
                                   std::vector<gpubuf>&  ibuffer,
                                   std::vector<void*>&   pibuffer,
                                   std::vector<void*>&   pobuffer)
    {
    }

    // finalize multi-GPU transform.  pobuffers are the pointers
    // provided to the FFT library's "execute" API.  obuffer is the
    // normal GPU buffer that is used for single-device FFTs.
    // gpu_output is the host buffer where transform output needs to
    // go for validation
    // Field is distributed among ranks and then among the number
    // of GPUs per rank (which defaults to 1).
    virtual void multi_gpu_finalize(std::vector<hostbuf>& gpu_output,
                                    std::vector<gpubuf>&  obuffer,
                                    std::vector<void*>&   pobuffer)
    {
    }

    // Create bricks in the specified field.  brick_grid has an
    // integer per dimension (batch and FFT dimensions), with the
    // number of bricks to split that dimension on.  Field length
    // starts with batch dimension, followed by FFT dimensions
    // slowest to fastest.
    // num_ranks represents the number of ranks used in the parallel
    // computer, which are assumed to have at least gpusperrank each
    void distribute_field(int                              gpusperrank,
                          const std::vector<unsigned int>& brick_grid,
                          std::vector<fft_field>&          fields,
                          const std::vector<size_t>&       field_length,
                          int                              num_ranks)
    {
        if(brick_grid.size() != field_length.size())
            throw std::runtime_error(
                "distribute field requires same number of dims for grid and field length");

        // if nothing's actually split, don't bother making bricks
        if(std::all_of(
               brick_grid.begin(), brick_grid.end(), [](const unsigned int g) { return g == 1; }))
            return;

        size_t total_bricks = product(brick_grid.begin(), brick_grid.end());

        auto& field = fields.emplace_back();

        // Start with empty brick in field
        field.bricks.reserve(total_bricks);
        field.bricks.emplace_back();

        // Go over the grid
        for(size_t i = 0; i < brick_grid.size(); ++i)
        {
            std::vector<fft_brick> cur_bricks;
            cur_bricks.swap(field.bricks);
            field.bricks.reserve(total_bricks);

            auto brick_count = brick_grid[i];
            auto cur_length  = field_length[i];

            // split current length, apply to all current bricks and
            // append bricks to field
            for(size_t ibrick = 0; ibrick < brick_count; ++ibrick)
            {
                for(const auto& b : cur_bricks)
                {
                    auto& new_brick = field.bricks.emplace_back(b);
                    new_brick.lower.push_back(cur_length / brick_count * ibrick);
                    // last brick needs to include the whole split len
                    if(ibrick == brick_count - 1)
                    {
                        new_brick.upper.push_back(cur_length);
                    }
                    else
                    {
                        new_brick.upper.push_back(std::min(
                            cur_length, new_brick.lower.back() + cur_length / brick_count));
                    }
                }
            }
        }

        // Give all bricks contiguous strides
        int brickIdx = 0;
        for(auto& b : field.bricks)
        {
            b.stride.resize(b.upper.size());

            // Fill strides from fastest to slowest
            size_t brick_dist = 1;
            for(size_t distIdx = 0; distIdx < b.upper.size(); ++distIdx)
            {
                *(b.stride.rbegin() + distIdx) = brick_dist;
                brick_dist *= *(b.upper.rbegin() + distIdx) - *(b.lower.rbegin() + distIdx);
            }

            // Split across ranks for a multi-process transform,
            // otherwise split across bricks.  assume there's one
            // rank/device per brick
            if(mp_lib == fft_mp_lib_none)
            {
                b.device = brickIdx;
            }
            else
            {
                int rank = brickIdx / gpusperrank; // determine MPI rank
                int gpu  = brickIdx % gpusperrank; // determine GPU within rank

                b.rank   = rank;
                b.device = gpu;
            }
            brickIdx++;
        }
    }

    // Distribute problem input among specified grid of devices/processors.
    // Grid specifies number of bricks per dimension, starting with batch
    // and ending with fastest FFT dimension. For single-proc single-proc
    // multi-gpu, gpusperrank represents the number of GPUs to use;
    // while for multi-proc it represents the number of GPUs on each rank.
    void distribute_input(int                              gpusperrank,
                          const std::vector<unsigned int>& brick_grid,
                          int                              num_ranks = 1)
    {
        auto len = length;
        len.insert(len.begin(), nbatch);
        distribute_field(gpusperrank, brick_grid, ifields, len, num_ranks);
    }

    // Distribute problem output among specified grid of devices/processors.
    // Grid specifies number of bricks per dimension, starting with batch
    // and ending with fastest FFT dimension.
    void distribute_output(int                              gpusperrank,
                           const std::vector<unsigned int>& brick_grid,
                           int                              num_ranks = 1)
    {
        auto len = olength();
        len.insert(len.begin(), nbatch);
        distribute_field(gpusperrank, brick_grid, ofields, len, num_ranks);
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
// command line and we store into an enum varaible
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

// Given a data type and precision, the distance between batches, and
// the batch size, allocate the required host buffer(s).
static std::vector<hostbuf> allocate_host_buffer(const fft_precision        precision,
                                                 const fft_array_type       type,
                                                 const std::vector<size_t>& size)
{
    std::vector<hostbuf> buffers(size.size());
    for(unsigned int i = 0; i < size.size(); ++i)
    {
        buffers[i].alloc(size[i] * var_size<size_t>(precision, type));
    }
    return buffers;
}

// Check if the required buffers fit in the device vram.
inline bool vram_fits_problem(const size_t prob_size, const size_t vram_avail, int deviceId = 0)
{
    // We keep a small margin of error for fitting the problem into vram:
    const size_t extra = 1 << 27;

    return vram_avail > prob_size + extra;
}

// Computes the twiddle table VRAM footprint for r2c/c2r transforms.
// This function will return 0 for the other transform types, since
// the VRAM footprint in rocFFT is negligible for the other cases.
inline size_t twiddle_table_vram_footprint(const fft_params& params)
{
    size_t vram_footprint = 0;

    // Add vram footprint from real/complex even twiddle buffer size.
    if(params.transform_type == fft_transform_type_real_forward
       || params.transform_type == fft_transform_type_real_inverse)
    {
        const auto realdim = params.length.back();
        if(realdim % 2 == 0)
        {
            const auto complex_size = params.precision == fft_precision_single ? 8 : 16;
            // even length twiddle size is 1/4 of the real size, but
            // in complex elements
            vram_footprint += realdim * complex_size / 4;
        }
    }

    return vram_footprint;
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
        size_t brick_size_bytes = compute_ptrdiff(brick->length(), brick->stride, 0, 0) * elem_size
                                  / (is_planar ? 2 : 1);
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
