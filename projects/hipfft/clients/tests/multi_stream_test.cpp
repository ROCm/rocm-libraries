// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights
// reserved.
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

#include "hipfft/hipfft.h"
#include <gtest/gtest.h>

#include "../../shared/arithmetic.h"
#include "../../shared/fft_params.h"
#include "../../shared/params_gen.h" // hash_prob
#include "../../shared/test_params.h" // externally-declared test parameters
#include "../hipfft_params.h"
#include <algorithm> // copy_n, any_of
#include <cmath> // M_PI, cos, sin
#include <exception>
#include <functional> // hasher
#include <iostream>
#include <limits> // numeric_limits
#include <random> // ranlux24_base, uniform_int_distribution
#include <sstream>
#include <stdexcept>
#ifdef WIN32
#include <synchapi.h> // Sleep
#else
#include <unistd.h> // usleep
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

DISABLE_WARNING_PUSH
DISABLE_WARNING_DEPRECATED_DECLARATIONS
DISABLE_WARNING_RETURN_TYPE
#include <hip/hip_runtime_api.h>
DISABLE_WARNING_POP

// a simple class capturing only the most elementary DFT-defining parameters
// for tetsing multi-stream DFT operations
class ParamsForMultiStreamDFT
{
    class unsupported_case : public std::runtime_error
    {
    public:
        unsupported_case(const std::string& msg)
            : std::runtime_error(msg)
        {
        }
    };

public:
    bool is_inverse() const
    {
        return transform_type != fft_transform_type_complex_forward
               && transform_type != fft_transform_type_real_forward;
    }
    size_t dim() const
    {
        return lengths.size();
    }

    bool is_real() const
    {
        return transform_type == fft_transform_type_real_inverse
               || transform_type == fft_transform_type_real_forward;
    }
    size_t real_scalar_type_size() const
    {
        size_t ret = 0;
        switch(precision)
        {
        case fft_precision_half:
            ret = sizeof(rocfft_fp16);
            break;
        case fft_precision_single:
            ret = sizeof(float);
            break;
        case fft_precision_double:
            ret = sizeof(double);
            break;
        default:
            throw std::runtime_error("Unknown precision");
            break;
        }
        return ret;
    }

    hipfft_params make_sub_dft_params(size_t step_id, size_t stream_id) const
    {
        if(step_id >= num_steps || stream_id >= num_streams)
        {
            throw std::domain_error(
                "step_id must be in (0, num_steps( and streamd_id must be in (0, num_streams( ");
        }
        hipfft_params sub_dft;
        size_t        total_batch_for_step = 0;
        if(step_id == is_inverse() ? 1 : 0)
        {
            // batched 1D DFTs
            sub_dft.length       = {lengths.back()};
            total_batch_for_step = product(lengths.begin(), lengths.end() - 1);
            // same placement, type of transform, precision as parent problem
            sub_dft.placement      = placement;
            sub_dft.transform_type = transform_type;
            sub_dft.precision      = precision;
            // default distances and strides are ok
        }
        else
        {
            // batched DFTs of dimension dim() - 1
            sub_dft.length.clear();
            std::copy_n(lengths.begin(), dim() - 1, std::back_inserter(sub_dft.length));
            total_batch_for_step = is_real() ? lengths.back() / 2 + 1 : lengths.back();
            // same precision as parent problem
            sub_dft.precision = precision;
            // always in-place
            sub_dft.placement = fft_placement_inplace;
            // always complex-to-complex:
            sub_dft.transform_type = is_inverse() ? fft_transform_type_complex_inverse
                                                  : fft_transform_type_complex_forward;
            // Distances and strides
            sub_dft.idist = sub_dft.odist = 1;
            std::vector<size_t> sub_strides(sub_dft.length.size());
            for(int stride_idx = sub_strides.size() - 1; stride_idx >= 0; stride_idx--)
            {
                if(stride_idx == sub_strides.size() - 1)
                {
                    sub_strides[stride_idx] = total_batch_for_step;
                }
                else
                {
                    sub_strides[stride_idx]
                        = sub_strides[stride_idx + 1] * sub_dft.length[stride_idx + 1];
                }
            }

            sub_dft.istride = sub_dft.ostride = sub_strides;
        }
        if(total_batch_for_step < num_streams)
        {
            throw unsupported_case("Stream-decomposition not implemented yet");
        }
        sub_dft.nbatch = total_batch_for_step / num_streams
                         + (stream_id < total_batch_for_step % num_streams ? 1 : 0);
        sub_dft.precision = precision;
        sub_dft.validate();
        return sub_dft;
    }

    std::vector<size_t>  lengths;
    fft_precision        precision      = fft_precision_single;
    fft_transform_type   transform_type = fft_transform_type_complex_forward;
    fft_result_placement placement      = fft_placement_inplace;

    bool is_supported() const
    {
        // check if division of work in num_streams streams is supported yet
        try
        {
            if(dim() < num_steps || dim() > 3
               || (std::any_of(lengths.begin(),
                               lengths.end(),
                               [](const decltype(lengths)::value_type& val) { return val <= 0; })))
            {
                throw unsupported_case("Invalid lengths");
            }
            if(precision != fft_precision_single && precision != fft_precision_double)
            {
                throw unsupported_case("Precision not supported");
            }
            if(transform_type != fft_transform_type_real_forward
               && transform_type != fft_transform_type_real_inverse
               && transform_type != fft_transform_type_complex_forward
               && transform_type != fft_transform_type_complex_inverse)
            {
                throw unsupported_case("Unknown transform type");
            }
            for(size_t step_id = 0; step_id < num_steps; step_id++)
            {
#ifdef __HIP_PLATFORM_NVIDIA__
                size_t stream_start_idx_real = 0;
#endif
                for(size_t stream_id = 0; stream_id < num_streams; stream_id++)
                {
#ifdef __HIP_PLATFORM_NVIDIA__
                    if(stream_start_idx_real % 2)
                        throw unsupported_case("Unaligned I/O data for cuFFT backend");
#endif

                    auto sub_dft = make_sub_dft_params(step_id, stream_id);
#ifdef __HIP_PLATFORM_NVIDIA__
                    if(sub_dft.is_real())
                    {
                        const auto real_dist = is_inverse() ? sub_dft.odist : sub_dft.idist;
                        stream_start_idx_real += real_dist * sub_dft.nbatch;
                    }
#endif
                }
            }
        }
        catch(const unsupported_case& e)
        {
            if(verbose > 0)
            {
                std::cout << e.what() << std::endl;
            }
            return false;
        }
        return true;
    }

    template <bool for_input_data>
    size_t get_data_size() const
    {
        size_t total_size = real_scalar_type_size();
        if(!is_real())
            total_size *= 2 * lengths.back();
        else
        {
            if((for_input_data == is_inverse()) || placement == fft_placement_inplace)
            {
                total_size *= 2 * (lengths.back() / 2 + 1);
            }
            else
                total_size *= lengths.back();
        }
        for(size_t length_idx = 0; length_idx < dim() - 1; length_idx++)
            total_size *= lengths[length_idx];
        return total_size;
    }

    template <bool for_input_data>
    std::vector<size_t> get_data_strides() const
    {
        std::vector<size_t> ret(1, 1);
        if(is_real())
        {
            if(for_input_data == is_inverse())
                ret.insert(ret.begin(), lengths.back() / 2 + 1);
            else
            {
                if(placement == fft_placement_inplace)
                    ret.insert(ret.begin(), 2 * (lengths.back() / 2 + 1));
                else
                    ret.insert(ret.begin(), lengths.back());
            }
        }
        else
            ret.insert(ret.begin(), lengths.back());
        for(size_t i = 2; i < dim(); i++)
        {
            ret.insert(ret.begin(), ret.front() * lengths[dim() - i]);
        }
        return ret;
    }

    std::string get_test_name() const
    {
        // use fft_params' token member function to generate unambiguous test names
        fft_params for_test_name;
        for_test_name.length         = lengths;
        for_test_name.precision      = precision;
        for_test_name.transform_type = transform_type;
        for_test_name.placement      = placement;
        for_test_name.validate();
        return for_test_name.token();
    }

    constexpr static size_t num_streams = 4;
    constexpr static size_t num_steps   = 2;
};

// Base gtest class for multi-strean tests.
class multiStreamTest : public ::testing::TestWithParam<ParamsForMultiStreamDFT>
{
protected:
    void SetUp() override {}
    void TearDown() override {}

public:
    static std::string TestName(const testing::TestParamInfo<multiStreamTest::ParamType>& info)
    {
        return info.param.get_test_name();
    }
};

template <typename real_data_type>
static void init_data(hostbuf&                       hostbuffer,
                      const ParamsForMultiStreamDFT& params,
                      const std::vector<size_t>&     harmonic)
{
    if(params.dim() < 2 || params.dim() > 3)
        throw std::invalid_argument("Only dimensions 2 and 3 can be considered");
    if(harmonic.size() != params.dim())
        throw std::invalid_argument(
            "As many harmonic components as the problem's dimension required");
    real_data_type*           data_ptr = static_cast<real_data_type*>(hostbuffer.data());
    const std::vector<size_t> strides  = params.get_data_strides<true /*input data*/>();

    auto phase = [](const std::vector<size_t>& k,
                    const std::vector<size_t>& h,
                    const std::vector<size_t>& l) {
        real_data_type ret = 0;
        for(size_t i = 0; i < k.size(); i++)
            ret += static_cast<real_data_type>((k[i] * h[i]) % l[i])
                   / static_cast<real_data_type>(l[i]);
        return 2.0 * M_PI * ret;
    };
    // set pre-factor to have clear unit spike(s) at targeted harmonic component(s) after transform
    real_data_type pre_factor = 1.0 / product(params.lengths.begin(), params.lengths.end());
    if(params.transform_type == fft_transform_type_real_forward)
    {
        auto length_it = params.lengths.begin();
        // multiply by 2 if the chosen harmonic component is not its own hermitian symmetric
        if(std::any_of(harmonic.begin(), harmonic.end(), [&](decltype(harmonic[0])& h) {
               return h != (*length_it - h) % *length_it++;
           }))
        {
            pre_factor *= 2.0;
        }
    }
    std::vector<size_t> multi_index(params.dim(), 0);
    for(size_t k = 0; k < (params.dim() == 2 ? 1 : params.lengths[0]); k++)
    {
        if(params.dim() > 2)
            multi_index[0] = k;
        for(size_t j = 0; j < params.lengths[params.dim() - 2]; j++)
        {
            multi_index[params.dim() - 2] = j;
            for(size_t i = 0; i < (params.transform_type == fft_transform_type_real_inverse
                                       ? params.lengths.back() / 2 + 1
                                       : params.lengths.back());
                i++)
            {
                const size_t data_idx = (params.dim() > 2 ? k * strides[0] : 0)
                                        + j * strides[params.dim() - 2]
                                        + i * strides[params.dim() - 1];
                multi_index[params.dim() - 1] = i;
                if(params.transform_type != fft_transform_type_real_forward)
                {
                    data_ptr[2 * data_idx]
                        = pre_factor * std::cos(phase(multi_index, harmonic, params.lengths));
                    data_ptr[2 * data_idx + 1]
                        = pre_factor * (params.is_inverse() ? -1.0 : +1.0)
                          * std::sin(phase(multi_index, harmonic, params.lengths));
                }
                else
                {
                    data_ptr[data_idx]
                        = pre_factor * std::cos(phase(multi_index, harmonic, params.lengths));
                }
            }
        }
    }
}

template <typename real_data_type>
static real_data_type max_error(const hostbuf&                 hostbuffer,
                                const ParamsForMultiStreamDFT& params,
                                const std::vector<size_t>&     harmonic)
{
    if(params.dim() < 2 || params.dim() > 3)
        throw std::invalid_argument("Only dimensions 2 and 3 can be considered");
    if(harmonic.size() != params.dim())
        throw std::invalid_argument(
            "As many harmonic components as the problem's dimension required");
    const real_data_type*     data_ptr = static_cast<real_data_type*>(hostbuffer.data());
    const std::vector<size_t> strides  = params.get_data_strides<false /*output data*/>();

    auto spike_expected = [&](const std::vector<size_t>& multi_index) {
        bool ret = multi_index == harmonic;
        if(!ret && params.transform_type == fft_transform_type_real_forward)
        {
            // could be the hermitian symmetry of the expected one
            auto length_it    = params.lengths.begin();
            auto multi_idx_it = multi_index.begin();
            ret = std::all_of(harmonic.begin(), harmonic.end(), [&](decltype(harmonic[0])& h) {
                return h == (*length_it - *multi_idx_it++) % *length_it++;
            });
        }
        return ret;
    };

    std::vector<size_t> multi_index(params.dim(), 0);
    real_data_type      max_abs_diff{0};
    for(size_t k = 0; k < (params.dim() == 2 ? 1 : params.lengths[0]); k++)
    {
        if(params.dim() > 2)
            multi_index[0] = k;
        for(size_t j = 0; j < params.lengths[params.dim() - 2]; j++)
        {
            multi_index[params.dim() - 2] = j;
            for(size_t i = 0; i < (params.transform_type == fft_transform_type_real_forward
                                       ? params.lengths.back() / 2 + 1
                                       : params.lengths.back());
                i++)
            {
                const size_t data_idx = (params.dim() > 2 ? k * strides[0] : 0)
                                        + j * strides[params.dim() - 2]
                                        + i * strides[params.dim() - 1];
                multi_index[params.dim() - 1]            = i;
                const real_data_type expected_real_value = spike_expected(multi_index) ? 1.0 : 0.0;
                if(params.transform_type != fft_transform_type_real_inverse)
                {
                    max_abs_diff = std::max(
                        max_abs_diff, std::fabs(data_ptr[2 * data_idx] - expected_real_value));
                    // imaginary part always expected to be 0
                    max_abs_diff = std::max(max_abs_diff, std::fabs(data_ptr[2 * data_idx + 1]));
                }
                else
                {
                    max_abs_diff = std::max(max_abs_diff,
                                            std::fabs(data_ptr[data_idx] - expected_real_value));
                }
            }
        }
    }
    return max_abs_diff;
}

template <typename T>
static void allocate_buffer(gpubuf_t<T>& buffer, const size_t desired_size)
{
    auto ret = buffer.alloc(desired_size);
    if(ret != hipSuccess)
    {
        n_hip_failures++;
        std::stringstream info;
        info << "Test failed to allocate " << desired_size << " bytes for gpu data";
        if(skip_runtime_fails)
        {
            GTEST_SKIP() << info.str();
        }
        else
        {
            GTEST_FAIL() << info.str();
        }
    }
}

TEST_P(multiStreamTest, impulseSignalOnOutput)
{
    fft_status                            fft_error_code    = fft_status_success;
    hipError_t                            hip_error_code    = hipSuccess;
    hipfftResult_t                        hipfft_error_code = HIPFFT_SUCCESS;
    std::stringstream                     info;
    ParamsForMultiStreamDFT               parameters = GetParam();
    std::hash<std::string>                hasher;
    std::ranlux24_base                    gen(random_seed + hasher(parameters.get_test_name()));
    std::uniform_int_distribution<size_t> harmonic_rng(std::numeric_limits<size_t>::min(),
                                                       std::numeric_limits<size_t>::max());

    if(!parameters.is_supported())
        GTEST_SKIP() << "Test not supported yet";
    // RAII encapsulation struct for hip streams:
    struct sub_dft_stream_t
    {
    private:
        bool sub_dft_is_done;
        bool cb_is_enqueued;

    public:
        struct init_failure : public std::exception
        {
        };
        hipStream_t hip_stream;
        sub_dft_stream_t()
            : sub_dft_is_done(false)
            , cb_is_enqueued(false)
        {
            auto ret = hipStreamCreate(&hip_stream);
            if(ret != hipSuccess)
                throw init_failure();
        }
        hipError_t enqueue_host_callback()
        {
            if(cb_is_enqueued)
                throw std::runtime_error("a callback is already enqueued for this stream");
            auto mark_stream_work_done_callback = [](hipStream_t, hipError_t, void* work_done_ptr) {
                *(static_cast<bool*>(work_done_ptr)) = true; // raise flag
            };
            const auto hip_status = hipStreamAddCallback(
                hip_stream, mark_stream_work_done_callback, &sub_dft_is_done, 0 /* must be 0 */);
            if(hip_status == hipSuccess)
                cb_is_enqueued = true;
            return hip_status;
        }

        ~sub_dft_stream_t()
        {
            if(cb_is_enqueued)
            {
                (void)hipStreamSynchronize(hip_stream);
            }
            (void)hipStreamDestroy(hip_stream);
        }

        bool done() const
        {
            return sub_dft_is_done;
        }
        void reset_flags()
        {
            sub_dft_is_done = cb_is_enqueued = false;
        }
    };
    // create the test streams
    std::vector<sub_dft_stream_t> test_streams;
    try
    {
        test_streams.resize(ParamsForMultiStreamDFT::num_streams);
    }
    catch(const sub_dft_stream_t::init_failure& e)
    {
        n_hip_failures++;
        info.str("");
        info << "Test failed to create " << ParamsForMultiStreamDFT::num_streams << " streams";
        if(skip_runtime_fails)
        {
            GTEST_SKIP() << info.str();
        }
        else
        {
            GTEST_FAIL() << info.str();
        }
    }
    catch(...)
    {
        GTEST_FAIL() << "Issue caught when creating " << ParamsForMultiStreamDFT::num_streams
                     << " streams";
    }
    // construct plans for each step and stream:
    hipfft_params sub_dft[ParamsForMultiStreamDFT::num_steps][ParamsForMultiStreamDFT::num_streams];
    try
    {
        for(size_t stream_id = 0; stream_id < ParamsForMultiStreamDFT::num_streams; stream_id++)
        {
            for(size_t step_id = 0; step_id < ParamsForMultiStreamDFT::num_steps; step_id++)
            {
                hipfft_params& dft_op = sub_dft[step_id][stream_id];

                dft_op         = parameters.make_sub_dft_params(step_id, stream_id);
                fft_error_code = dft_op.create_plan();
                if(fft_error_code != fft_status_success)
                {
                    GTEST_FAIL() << "Failed to create hipfft plan for step id " << step_id
                                 << ", stream id " << stream_id
                                 << " (sub-DFT token : " << dft_op.token() << ")";
                }
                hipfft_error_code = dft_op.set_stream(test_streams[stream_id].hip_stream);
                if(hipfft_error_code != HIPFFT_SUCCESS)
                {
                    GTEST_FAIL() << "Failed to set stream for step id " << step_id << ", stream id "
                                 << stream_id;
                }
                if(verbose > 1)
                {
                    std::cout << "token of sub-DFT created for step id " << step_id
                              << " and stream id " << stream_id << ": " << dft_op.token()
                              << std::endl;
                }
            }
        }
    }
    catch(fft_params::work_buffer_alloc_failure& e)
    {
        info.str("");
        info << "Allocation failure detected during the creation of the sub-DFT plans";
        ++n_hip_failures;
        if(skip_runtime_fails)
        {
            GTEST_SKIP() << info.str();
        }
        else
        {
            GTEST_FAIL() << info.str();
        }
    }
    catch(...)
    {
        GTEST_FAIL() << "The test failed to create the required sub-DFT plans";
    }
    // compute the full DFT described by parameters by decomposing work in
    // ParamsForMultiStreamDFT::num_streams different streams
    const size_t isize = parameters.get_data_size<true>();
    const size_t osize = parameters.get_data_size<false>();
    gpubuf_t     input_buf, output_buf;
    allocate_buffer(input_buf, isize);
    if(parameters.placement != fft_placement_inplace)
    {
        allocate_buffer(output_buf, osize);
        // avoid false negatives via nonzero initialization:
        hip_error_code = hipMemset(output_buf.data(), 1, osize);
        if(hip_error_code != hipSuccess)
        {
            GTEST_FAIL() << "Non-zero initialization of output buffer failed";
        }
    }

    hostbuf hostbuffer;
    try
    {
        hostbuffer.alloc(std::max(isize, osize));
    }
    catch(HOSTBUF_MEM_USAGE& e)
    {
        info.str("");
        info << "could not allocate host buffer";
        GTEST_SKIP() << info.str();
    }

    std::vector<size_t> expected_harmonic(parameters.dim());
    for(size_t i = 0; i < parameters.dim(); i++)
    {
        const size_t max_harmonic
            = i == parameters.dim() - 1
                      && parameters.transform_type == fft_transform_type_real_forward
                  ? parameters.lengths[i] / 2 + 1
                  : parameters.lengths[i];

        expected_harmonic[i] = harmonic_rng(gen) % max_harmonic;
        if(verbose > 0)
        {
            std::cout << "Chosen harmonic component: " << expected_harmonic[i] << std::endl;
        }
    }

    if(parameters.precision == fft_precision_double)
        init_data<double>(hostbuffer, parameters, expected_harmonic);
    else
        init_data<float>(hostbuffer, parameters, expected_harmonic);

    // data copy to device (synchronizing)
    hip_error_code = hipMemcpy(input_buf.data(), hostbuffer.data(), isize, hipMemcpyHostToDevice);
    if(hip_error_code != hipSuccess)
    {
        n_hip_failures++;
        info.str("");
        info << "Test failed to copy initialized data set from host to device";
        if(skip_runtime_fails)
        {
            GTEST_SKIP() << info.str();
        }
        else
        {
            GTEST_FAIL() << info.str();
        }
    }
    // Computation of the full DFT described by parameters in 2 steps, each involving
    // ParamsForMultiStreamDFT::num_streams different streams. Synchronization done via
    // a (host) callback invoked upon completion of every stream's task.
    void* step_stream_input_data_ptr  = nullptr;
    void* step_stream_output_data_ptr = nullptr;
    for(size_t step_id = 0; step_id < ParamsForMultiStreamDFT::num_steps; step_id++)
    {
        const size_t elementary_itype_size
            = parameters.real_scalar_type_size()
              * (parameters.transform_type == fft_transform_type_real_forward && step_id == 0 ? 1
                                                                                              : 2);
        const size_t elementary_otype_size
            = parameters.real_scalar_type_size()
              * (parameters.transform_type == fft_transform_type_real_inverse && step_id == 1 ? 1
                                                                                              : 2);
        if(parameters.placement == fft_placement_inplace)
        {
            // every input/output data for the subproblems is in input_buf
            step_stream_input_data_ptr  = input_buf.data();
            step_stream_output_data_ptr = input_buf.data();
        }
        else
        {
            // stream's input/output data depends on step and type of global problem
            step_stream_input_data_ptr
                = step_id == 0 || parameters.is_inverse() ? input_buf.data() : output_buf.data();
            step_stream_output_data_ptr
                = step_id == 1 || !parameters.is_inverse() ? output_buf.data() : input_buf.data();
        }
        for(size_t stream_id = 0; stream_id < ParamsForMultiStreamDFT::num_streams; stream_id++)
        {
            hipfft_params& dft_op = sub_dft[step_id][stream_id];
            fft_error_code
                = dft_op.execute(step_stream_input_data_ptr, step_stream_output_data_ptr);
            if(fft_error_code != fft_status_success)
            {
                GTEST_FAIL() << "execution failed for step id " << step_id << " and stream id "
                             << stream_id;
            }

            hip_error_code = test_streams[stream_id].enqueue_host_callback();
            if(hip_error_code != hipSuccess)
            {
                n_hip_failures++;
                info.str("");
                info << "Test failed to add callback function forstep id " << step_id
                     << " and stream id " << stream_id;
                if(skip_runtime_fails)
                {
                    GTEST_SKIP() << info.str();
                }
                else
                {
                    GTEST_FAIL() << info.str();
                }
            }
            // increment stream data pointers for next submissions
            step_stream_input_data_ptr = static_cast<char*>(step_stream_input_data_ptr)
                                         + dft_op.nbatch * dft_op.idist * elementary_itype_size;
            step_stream_output_data_ptr = static_cast<char*>(step_stream_output_data_ptr)
                                          + dft_op.nbatch * dft_op.odist * elementary_otype_size;
        }
        // Check if the callbacks get invoked within 10 s. If not, the stream set above was likely
        // ignored in the sub_dft operations (integer "times" in us below)
        size_t           time_waited_us            = 0;
        constexpr size_t sleep_time_us             = 1000; // 1 ms
        constexpr size_t failure_time_threshold_us = 10000000; // 10^7 us := 10 s
        while(std::any_of(test_streams.begin(),
                          test_streams.end(),
                          [](const sub_dft_stream_t& stream) { return !stream.done(); })
              && time_waited_us <= failure_time_threshold_us)
        {
#ifdef WIN32
            Sleep(sleep_time_us / 1000); // argument in ms
#else
            usleep(sleep_time_us);
#endif
            time_waited_us += sleep_time_us;
        }
        if(time_waited_us > failure_time_threshold_us)
        {
            // The added callback probably was never invoked, i.e., the above set_stream
            // was not taken into consideration by some sub_dft plan.
            GTEST_FAIL() << "Time limit exceeded";
        }
        else
        {
            for(auto& stream : test_streams)
            {
                stream.reset_flags();
            }
        }
    }
    // verify results:
    hip_error_code = hipMemcpy(
        hostbuffer.data(),
        (parameters.placement == fft_placement_inplace ? input_buf.data() : output_buf.data()),
        osize,
        hipMemcpyDeviceToHost);
    if(hip_error_code != hipSuccess)
    {
        n_hip_failures++;
        info.str("");
        info << "Test failed to copy results from device back to host";
        if(skip_runtime_fails)
        {
            GTEST_SKIP() << info.str();
        }
        else
        {
            GTEST_FAIL() << info.str();
        }
    }
    // always using doubles for measured max error and error thresholds for convenience (no data loss)
    const double error_threshold
        = (parameters.precision == fft_precision_single ? single_epsilon : double_epsilon)
          * log(product(parameters.lengths.begin(), parameters.lengths.end()));
    double measured_max_error = 0.0;
    if(parameters.precision == fft_precision_single)
        measured_max_error = max_error<float>(hostbuffer, parameters, expected_harmonic);
    else
        measured_max_error = max_error<double>(hostbuffer, parameters, expected_harmonic);

    ASSERT_LE(measured_max_error, error_threshold);
}

static std::vector<ParamsForMultiStreamDFT>
    generate_full_scope_for(const std::vector<std::vector<size_t>>& set_of_test_lengths)
{
    std::vector<ParamsForMultiStreamDFT> ret;
    ParamsForMultiStreamDFT              to_add;
    // set_of_lengths assumed not to contain duplicates
    for(const auto& test_lengths : set_of_test_lengths)
    {
        to_add.lengths = test_lengths;
        for(auto type : {fft_transform_type_complex_forward,
                         fft_transform_type_real_forward,
                         fft_transform_type_complex_inverse,
                         fft_transform_type_real_inverse})
        {
            to_add.transform_type = type;
            for(auto prec : {fft_precision_single, fft_precision_double})
            {
                to_add.precision = prec;
                for(auto place : {fft_placement_inplace, fft_placement_notinplace})
                {
                    to_add.placement = place;
                    if(!to_add.is_supported())
                        continue;

                    const double roll     = hash_prob(random_seed, to_add.get_test_name());
                    const double run_prob = test_prob * (to_add.is_real() ? real_prob_factor : 1.0);

                    if(roll > run_prob)
                    {
                        if(verbose > 4)
                        {
                            std::cout << "Test skipped: (roll=" << roll << " > " << run_prob
                                      << ")\n";
                        }
                        continue;
                    }
                    ret.push_back(to_add);
                }
            }
        }
    }
    return ret;
}

// note: generate_full_scope_for will create 16 instance for every created size
// --> 16*set_size tests generated in the end
template <size_t max_double_data_byte_size, size_t set_size = 32>
static std::vector<std::vector<size_t>> create_random_set_of_sizes()
{
    // limiting to lengths <= 512 per dimension
    std::ranlux24_base                    gen(random_seed);
    std::uniform_int_distribution<size_t> size_rng(3 * ParamsForMultiStreamDFT::num_streams, 512);
    // lexicographically-sorted set to return
    std::vector<std::vector<size_t>> ret;
    while(ret.size() < set_size)
    {
        // alternate between 2D and 3D sizes
        const size_t        dim = ret.size() % 2 == 0 ? 2 : 3;
        std::vector<size_t> to_add(dim, 0);
        for(auto& length : to_add)
            length = size_rng(gen);
        if(2 * product(to_add.begin(), to_add.end()) * sizeof(double) > max_double_data_byte_size)
            continue;
        auto it = std::lower_bound(ret.begin(), ret.end(), to_add);
        if(it == ret.end() || *it != to_add)
        {
            ret.insert(it, to_add);
        }
    }
    return ret;
}

static constexpr size_t max_byte_size = 128 * 1024 * 1024; // limit data sets to 128 MiB max

INSTANTIATE_TEST_SUITE_P(
    StreamDivision,
    multiStreamTest,
    ::testing::ValuesIn(generate_full_scope_for(create_random_set_of_sizes<max_byte_size>())),
    multiStreamTest::TestName);

// The list of test parameters dynamically generated in the instantiation above may be empty
// if low test probabilities are used. The following ensures such cases do not make gtest
// report an error due to uninstantiated multiStreamTest, e.g., with option smoketest.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(multiStreamTest);
