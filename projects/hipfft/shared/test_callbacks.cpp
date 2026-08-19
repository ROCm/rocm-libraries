// Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "test_callbacks.h"
#include "hip/hiprtc.h"
#include "rocfft_complex.h"
#include <string>

#ifdef ROCFFT_MPI_ENABLE
#include <mpi.h>
#endif

// load/store callbacks - cbdata in each is actually a scalar double
// with a number to apply to each element
template <typename Tdata>
__host__ __device__ Tdata load_callback(Tdata* input, size_t offset, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<const callback_test_data*>(cbdata);
    // multiply each element by scalar
    return input[offset] * static_cast<decltype(std::real(input[offset]))>(testdata->scalar);
}

static const char* load_callback_jit = R"(
CALLBACK_LINKAGE
__device__ Tdata load_callback(void* input_void, unsigned long long offset, void* cbdata, void* sharedMem)
{
    static_assert(sizeof(size_t) == sizeof(unsigned long long));
    auto input = static_cast<const Trocfft*>(input_void);
    auto testdata = static_cast<const callback_test_data*>(cbdata);
    // multiply each element by scalar
    return element_convert<Tdata, Trocfft>(input[offset] * static_cast<Treal>(testdata->scalar));
}
)";

__device__ auto load_callback_dev_half           = load_callback<rocfft_fp16>;
__device__ auto load_callback_dev_complex_half   = load_callback<rocfft_complex<rocfft_fp16>>;
__device__ auto load_callback_dev_float          = load_callback<float>;
__device__ auto load_callback_dev_complex_float  = load_callback<rocfft_complex<float>>;
__device__ auto load_callback_dev_double         = load_callback<double>;
__device__ auto load_callback_dev_complex_double = load_callback<rocfft_complex<double>>;

// load/store callbacks - cbdata in each is actually a scalar double
// with a number to apply to each element
template <typename Tdata>
__host__ __device__ Tdata
    load_callback_round_trip_inverse(Tdata* input, size_t offset, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<const callback_test_data*>(cbdata);
    // subtract each element by scalar
    return input[offset] - static_cast<decltype(std::real(input[offset]))>(testdata->scalar);
}

static const char* load_callback_round_trip_inverse_jit = R"(
CALLBACK_LINKAGE
__device__ Tdata
    load_callback_round_trip_inverse(void* input_void, unsigned long long offset, void* cbdata, void* sharedMem)
{
    static_assert(sizeof(size_t) == sizeof(unsigned long long));
    auto input = static_cast<const Trocfft*>(input_void);
    auto testdata = static_cast<const callback_test_data*>(cbdata);
    // subtract each element by scalar
    return element_convert<Tdata, Trocfft>(input[offset] - static_cast<Treal>(testdata->scalar));
}
)";

__device__ auto load_callback_round_trip_inverse_dev_half
    = load_callback_round_trip_inverse<rocfft_fp16>;
__device__ auto load_callback_round_trip_inverse_dev_complex_half
    = load_callback_round_trip_inverse<rocfft_complex<rocfft_fp16>>;
__device__ auto load_callback_round_trip_inverse_dev_float
    = load_callback_round_trip_inverse<float>;
__device__ auto load_callback_round_trip_inverse_dev_complex_float
    = load_callback_round_trip_inverse<rocfft_complex<float>>;
__device__ auto load_callback_round_trip_inverse_dev_double
    = load_callback_round_trip_inverse<double>;
__device__ auto load_callback_round_trip_inverse_dev_complex_double
    = load_callback_round_trip_inverse<rocfft_complex<double>>;

void* get_load_callback_funcptr(fft_array_type itype,
                                fft_precision  precision,
                                bool           round_trip_inverse)
{
    void*      load_callback_host = nullptr;
    hipError_t hip_status         = hipErrorUnknown;
    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case fft_precision_half:
            if(round_trip_inverse)
            {
                hip_status = hipMemcpyFromSymbol(
                    &load_callback_host,
                    HIP_SYMBOL(load_callback_round_trip_inverse_dev_complex_half),
                    sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            else
            {
                hip_status = hipMemcpyFromSymbol(
                    &load_callback_host, HIP_SYMBOL(load_callback_dev_complex_half), sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            return load_callback_host;
        case fft_precision_single:
            if(round_trip_inverse)
            {
                hip_status = hipMemcpyFromSymbol(
                    &load_callback_host,
                    HIP_SYMBOL(load_callback_round_trip_inverse_dev_complex_float),
                    sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            else
            {
                hip_status = hipMemcpyFromSymbol(&load_callback_host,
                                                 HIP_SYMBOL(load_callback_dev_complex_float),
                                                 sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            return load_callback_host;
        case fft_precision_double:
            if(round_trip_inverse)
            {
                hip_status = hipMemcpyFromSymbol(
                    &load_callback_host,
                    HIP_SYMBOL(load_callback_round_trip_inverse_dev_complex_double),
                    sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            else
            {
                hip_status = hipMemcpyFromSymbol(&load_callback_host,
                                                 HIP_SYMBOL(load_callback_dev_complex_double),
                                                 sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            return load_callback_host;
        }
    }
    case fft_array_type_real:
    {
        switch(precision)
        {
        case fft_precision_half:
            if(round_trip_inverse)
            {
                hip_status
                    = hipMemcpyFromSymbol(&load_callback_host,
                                          HIP_SYMBOL(load_callback_round_trip_inverse_dev_half),
                                          sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            else
            {
                hip_status = hipMemcpyFromSymbol(
                    &load_callback_host, HIP_SYMBOL(load_callback_dev_half), sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            return load_callback_host;
        case fft_precision_single:
            if(round_trip_inverse)
            {
                hip_status
                    = hipMemcpyFromSymbol(&load_callback_host,
                                          HIP_SYMBOL(load_callback_round_trip_inverse_dev_float),
                                          sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            else
            {
                hip_status = hipMemcpyFromSymbol(
                    &load_callback_host, HIP_SYMBOL(load_callback_dev_float), sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            return load_callback_host;
        case fft_precision_double:
            if(round_trip_inverse)
            {
                hip_status
                    = hipMemcpyFromSymbol(&load_callback_host,
                                          HIP_SYMBOL(load_callback_round_trip_inverse_dev_double),
                                          sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            else
            {
                hip_status = hipMemcpyFromSymbol(
                    &load_callback_host, HIP_SYMBOL(load_callback_dev_double), sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }

            return load_callback_host;
        }
    }
    default:
        // planar is unsupported for now
        return load_callback_host;
    }
}

// get declarations used by JIT callbacks:
// - Tdata: data type in the function signature, e.g. hipfftComplex
// - Treal: data type of a real value
// - Trocfft: rocfft_complex version of Tdata if complex, otherwise == Tdata
// - element_convert: templated helper function to construct a Tdata from a Trocfft, or vice-versa
static std::string get_jit_callback_decls(fft_array_type itype, fft_precision precision)
{
    std::string ret = "#ifdef __HIP_PLATFORM_AMD__\n"
                      "// rocFFT does not mangle the provided symbol names, so give\n"
                      "// the callback C linkage to effectively disable mangling.\n"
                      "#define CALLBACK_LINKAGE extern \"C\"\n"
                      "typedef hipComplex callbackFloatComplex;\n"
                      "typedef hipDoubleComplex callbackDoubleComplex;\n"
                      "#else\n"
                      "#define CALLBACK_LINKAGE\n"
                      "// cuFFT callbacks require CUDA complex types to match the\n"
                      "// expected function signature exactly\n"
                      "#include <cuComplex.h>\n"
                      "typedef cuComplex callbackFloatComplex;\n"
                      "typedef cuDoubleComplex callbackDoubleComplex;\n"
                      "#endif\n";
    switch(itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case fft_precision_half:
            ret += "typedef rocfft_complex<rocfft_fp16> Tdata;"
                   "typedef rocfft_fp16 Treal;"
                   "typedef rocfft_complex<rocfft_fp16> Trocfft;";
            break;
        case fft_precision_single:
            ret += "typedef callbackFloatComplex Tdata;"
                   "typedef float Treal;"
                   "typedef rocfft_complex<float> Trocfft;";
            break;
        case fft_precision_double:
            ret += "typedef callbackDoubleComplex Tdata;"
                   "typedef double Treal;"
                   "typedef rocfft_complex<double> Trocfft;";
            break;
        }
        ret += "template<typename Tdest,typename Tsrc>  __device__ Tdest element_convert(Tsrc "
               "src) { "
               "return {src.x, src.y}; "
               "}";
        break;
    }
    case fft_array_type_real:
    {
        switch(precision)
        {
        case fft_precision_half:
            ret += "typedef rocfft_fp16 Tdata;"
                   "typedef rocfft_fp16 Treal;"
                   "typedef rocfft_fp16 Trocfft;";
            break;
        case fft_precision_single:
            ret += "typedef float Tdata;"
                   "typedef float Treal;"
                   "typedef float Trocfft;";
            break;
        case fft_precision_double:
            ret += "typedef double Tdata;"
                   "typedef double Treal;"
                   "typedef double Trocfft;";
            break;
        }
        ret += "template<typename Tdest,typename Tsrc> __device__ Tdest element_convert(Tsrc "
               "src) { "
               "return src; }";
        break;
    }
    default:
        throw std::runtime_error("planar callbacks are unsupported");
    }
    return ret;
}

std::vector<char> compile_jit_callback(const std::string& src)
{
    struct RaiiState
    {
        hiprtcProgram prog = nullptr;
        ~RaiiState()
        {
            if(prog)
            {
                hiprtcDestroyProgram(&prog);
            }
        }
    };
    RaiiState state;

    auto err
        = hiprtcCreateProgram(&state.prog, src.c_str(), "rocfft_callback.hip", 0, nullptr, nullptr);
    if(err != HIPRTC_SUCCESS)
    {
        throw hiprtc_runtime_error{"unable to create program", err};
    }

    std::vector<const char*> options;
#ifdef __HIP_PLATFORM_AMD__
    options.push_back("-O3");
    options.push_back("--offload-arch=amdgcnspirv");
#else
#ifdef HIPFFT_CUDA_INCLUDE
    options.push_back("-I" HIPFFT_CUDA_INCLUDE);
#endif
    options.push_back("-dlto");
    options.push_back("--relocatable-device-code=true");
#endif

    err = hiprtcCompileProgram(state.prog, options.size(), options.data());
    if(err != HIPRTC_SUCCESS)
    {
        size_t logSize = 0;
        hiprtcGetProgramLogSize(state.prog, &logSize);

        if(logSize)
        {
            std::vector<char> log(logSize, '\0');
            if(hiprtcGetProgramLog(state.prog, log.data()) == HIPRTC_SUCCESS)
                throw hiprtc_runtime_error{std::string(log.begin(), log.end()), err};
        }
        throw hiprtc_runtime_error{"compile failed without log", err};
    }

    size_t            codeSize;
    std::vector<char> code;
#ifdef __HIP_PLATFORM_AMD__
    err = hiprtcGetBitcodeSize(state.prog, &codeSize);
    if(err != HIPRTC_SUCCESS)
        throw hiprtc_runtime_error{"failed to get bitcode size", err};

    code.resize(codeSize);
    err = hiprtcGetBitcode(state.prog, code.data());
    if(err != HIPRTC_SUCCESS)
        throw hiprtc_runtime_error{"failed to get bitcode", err};
#else
    auto nverr = nvrtcGetLTOIRSize(state.prog, &codeSize);
    if(nverr != NVRTC_SUCCESS)
        throw hiprtc_runtime_error{"failed to get bitcode size", nvrtcResultTohiprtcResult(nverr)};

    code.resize(codeSize);
    nverr = nvrtcGetLTOIR(state.prog, code.data());
    if(nverr != NVRTC_SUCCESS)
        throw hiprtc_runtime_error{"failed to get bitcode", nvrtcResultTohiprtcResult(nverr)};
#endif
    return code;
}

extern const char* rocfft_complex_h;
std::vector<char>
    get_load_callback_jit(fft_array_type itype, fft_precision precision, bool round_trip_inverse)
{
    std::string src = rocfft_complex_h;
    src += get_jit_callback_decls(itype, precision);
    src += callback_test_data_jit;

    src += round_trip_inverse ? load_callback_round_trip_inverse_jit : load_callback_jit;

    // compile to spirv
    return compile_jit_callback(src);
}

template <typename Tdata>
__host__ __device__ static void
    store_callback(Tdata* output, size_t offset, Tdata element, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<callback_test_data*>(cbdata);
    // add scalar to each element
    output[offset] = element + static_cast<decltype(std::real(output[offset]))>(testdata->scalar);
}

static const char* store_callback_jit = R"(
CALLBACK_LINKAGE
__device__ void
    store_callback(void* output_void, unsigned long long offset, Tdata element, void* cbdata, void* sharedMem)
{
    static_assert(sizeof(size_t) == sizeof(unsigned long long));
    auto output = static_cast<Trocfft*>(output_void);
    auto testdata = static_cast<callback_test_data*>(cbdata);
    // add scalar to each element
    output[offset] = element_convert<Trocfft,Tdata>(element) + static_cast<Treal>(testdata->scalar);
}
)";

__device__ auto store_callback_dev_half           = store_callback<rocfft_fp16>;
__device__ auto store_callback_dev_complex_half   = store_callback<rocfft_complex<rocfft_fp16>>;
__device__ auto store_callback_dev_float          = store_callback<float>;
__device__ auto store_callback_dev_complex_float  = store_callback<rocfft_complex<float>>;
__device__ auto store_callback_dev_double         = store_callback<double>;
__device__ auto store_callback_dev_complex_double = store_callback<rocfft_complex<double>>;

template <typename Tdata>
__host__ __device__ static void store_callback_round_trip_inverse(
    Tdata* output, size_t offset, Tdata element, void* cbdata, void* sharedMem)
{
    auto testdata = static_cast<callback_test_data*>(cbdata);
    // divide each element by scalar
    output[offset] = element / static_cast<decltype(std::real(output[offset]))>(testdata->scalar);
}
__device__ auto store_callback_round_trip_inverse_dev_half
    = store_callback_round_trip_inverse<rocfft_fp16>;
__device__ auto store_callback_round_trip_inverse_dev_complex_half
    = store_callback_round_trip_inverse<rocfft_complex<rocfft_fp16>>;
__device__ auto store_callback_round_trip_inverse_dev_float
    = store_callback_round_trip_inverse<float>;
__device__ auto store_callback_round_trip_inverse_dev_complex_float
    = store_callback_round_trip_inverse<rocfft_complex<float>>;
__device__ auto store_callback_round_trip_inverse_dev_double
    = store_callback_round_trip_inverse<double>;
__device__ auto store_callback_round_trip_inverse_dev_complex_double
    = store_callback_round_trip_inverse<rocfft_complex<double>>;

static const char* store_callback_round_trip_inverse_jit = R"(
CALLBACK_LINKAGE
__device__ void store_callback_round_trip_inverse(
    void* output_void, unsigned long long offset, Tdata element, void* cbdata, void* sharedMem)
{
    static_assert(sizeof(size_t) == sizeof(unsigned long long));
    auto output = static_cast<Trocfft*>(output_void);
    auto testdata = static_cast<callback_test_data*>(cbdata);
    // divide each element by scalar
    output[offset] = element_convert<Trocfft,Tdata>(element) / static_cast<Treal>(testdata->scalar);
}
)";

void* get_store_callback_funcptr(fft_array_type otype,
                                 fft_precision  precision,
                                 bool           round_trip_inverse)
{
    void*      store_callback_host = nullptr;
    hipError_t hip_status          = hipErrorUnknown;
    switch(otype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(precision)
        {
        case fft_precision_half:
            if(round_trip_inverse)
            {
                hip_status = hipMemcpyFromSymbol(
                    &store_callback_host,
                    HIP_SYMBOL(store_callback_round_trip_inverse_dev_complex_half),
                    sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            else
            {
                hip_status = hipMemcpyFromSymbol(&store_callback_host,
                                                 HIP_SYMBOL(store_callback_dev_complex_half),
                                                 sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            return store_callback_host;
        case fft_precision_single:
            if(round_trip_inverse)
            {
                hip_status = hipMemcpyFromSymbol(
                    &store_callback_host,
                    HIP_SYMBOL(store_callback_round_trip_inverse_dev_complex_float),
                    sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            else
            {
                hip_status = hipMemcpyFromSymbol(&store_callback_host,
                                                 HIP_SYMBOL(store_callback_dev_complex_float),
                                                 sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            return store_callback_host;
        case fft_precision_double:
            if(round_trip_inverse)
            {
                hip_status = hipMemcpyFromSymbol(
                    &store_callback_host,
                    HIP_SYMBOL(store_callback_round_trip_inverse_dev_complex_double),
                    sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            else
            {
                hip_status = hipMemcpyFromSymbol(&store_callback_host,
                                                 HIP_SYMBOL(store_callback_dev_complex_double),
                                                 sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            return store_callback_host;
        }
    }
    case fft_array_type_real:
    {
        switch(precision)
        {
        case fft_precision_half:
            if(round_trip_inverse)
            {
                hip_status
                    = hipMemcpyFromSymbol(&store_callback_host,
                                          HIP_SYMBOL(store_callback_round_trip_inverse_dev_half),
                                          sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            else
            {
                hip_status = hipMemcpyFromSymbol(
                    &store_callback_host, HIP_SYMBOL(store_callback_dev_half), sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            return store_callback_host;
        case fft_precision_single:
            if(round_trip_inverse)
            {
                hip_status
                    = hipMemcpyFromSymbol(&store_callback_host,
                                          HIP_SYMBOL(store_callback_round_trip_inverse_dev_float),
                                          sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            else
            {
                hip_status = hipMemcpyFromSymbol(
                    &store_callback_host, HIP_SYMBOL(store_callback_dev_float), sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            return store_callback_host;
        case fft_precision_double:
            if(round_trip_inverse)
            {
                hip_status
                    = hipMemcpyFromSymbol(&store_callback_host,
                                          HIP_SYMBOL(store_callback_round_trip_inverse_dev_double),
                                          sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            else
            {
                hip_status = hipMemcpyFromSymbol(
                    &store_callback_host, HIP_SYMBOL(store_callback_dev_double), sizeof(void*));
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipMemcpyFromSymbol failed", hip_status);
            }
            return store_callback_host;
        }
    }
    default:
        // planar is unsupported for now
        return store_callback_host;
    }
}

std::vector<char>
    get_store_callback_jit(fft_array_type otype, fft_precision precision, bool round_trip_inverse)
{
    std::string src = rocfft_complex_h;
    src += get_jit_callback_decls(otype, precision);
    src += callback_test_data_jit;

    src += round_trip_inverse ? store_callback_round_trip_inverse_jit : store_callback_jit;

    // compile to spirv
    return compile_jit_callback(src);
}

// Apply store callback if necessary
void apply_store_callback(const fft_params& params, std::vector<hostbuf>& output)
{
    if(params.run_callbacks == fft_callback_type_none)
        return;

    callback_test_data cbdata;
    cbdata.scalar = params.store_cb_scalar;

    switch(params.otype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(rocfft_complex<rocfft_fp16>);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin
                = reinterpret_cast<rocfft_complex<rocfft_fp16>*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(rocfft_complex<float>);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<rocfft_complex<float>*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(rocfft_complex<double>);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<rocfft_complex<double>*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        }
    }
    break;
    case fft_array_type_complex_planar:
    case fft_array_type_hermitian_planar:
    {
        throw std::runtime_error("planar callbacks are not supported");
    }
    break;
    case fft_array_type_real:
    {
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(rocfft_fp16);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<rocfft_fp16*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(float);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<float*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                store_callback(output_begin, i, element, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(double);
            const size_t num_elems = output.front().size() / elem_size;

            auto output_begin = reinterpret_cast<double*>(output.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                auto& element = output_begin[i];
                store_callback(output_begin, i, element, &cbdata, nullptr);
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
}

// apply load callback if necessary
void apply_load_callback(const fft_params& params, std::vector<hostbuf>& input)
{
    if(params.run_callbacks == fft_callback_type_none)
        return;
    // we're applying callbacks to FFTW input/output which we can
    // assume is contiguous and non-planar

    callback_test_data cbdata;
    cbdata.scalar = params.load_cb_scalar;

    switch(params.itype)
    {
    case fft_array_type_complex_interleaved:
    case fft_array_type_hermitian_interleaved:
    {
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(rocfft_complex<rocfft_fp16>);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<rocfft_complex<rocfft_fp16>*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(rocfft_complex<float>);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<rocfft_complex<float>*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(rocfft_complex<double>);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<rocfft_complex<double>*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        }
    }
    break;
    case fft_array_type_real:
    {
        switch(params.precision)
        {
        case fft_precision_half:
        {
            const size_t elem_size = sizeof(rocfft_fp16);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<rocfft_fp16*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_single:
        {
            const size_t elem_size = sizeof(float);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<float*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
            }
            break;
        }
        case fft_precision_double:
        {
            const size_t elem_size = sizeof(double);
            const size_t num_elems = input.front().size() / elem_size;

            auto input_begin = reinterpret_cast<double*>(input.front().data());
            for(size_t i = 0; i < num_elems; ++i)
            {
                input_begin[i] = load_callback(input_begin, i, &cbdata, nullptr);
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
}

// For the current rank, get a vector of load callback function +
// data pointers.  The pointers need to be in the order that
// fields+bricks were specified to the FFT plan.  Pointers need to be
// copied to the host from the device specified by the respective
// brick.
void get_rank_load_callbacks_funcptr(const fft_params&                          params,
                                     std::vector<void*>&                        load_cb_func,
                                     std::vector<void*>&                        load_cb_data,
                                     bool                                       round_trip_inverse,
                                     std::vector<gpubuf_t<callback_test_data>>& all_cb_data)
{
    int mpi_rank = params.get_process_rank();

    // Copy callback pointer from current device and add to output vec
    auto add_load_cb = [&]() {
        void* load_cb_host
            = get_load_callback_funcptr(params.itype, params.precision, round_trip_inverse);

        callback_test_data load_cb_data_host;

        if(round_trip_inverse)
        {
            load_cb_data_host.scalar = params.store_cb_scalar;
        }
        else
        {
            load_cb_data_host.scalar = params.load_cb_scalar;
        }

        auto& load_cb_data_dev = all_cb_data.emplace_back();
        auto  hip_status       = load_cb_data_dev.alloc(sizeof(callback_test_data));
        if(hip_status != hipSuccess)
        {
            throw hip_runtime_error(
                "Error occurred when allocating device memory for loading callback", hip_status);
        }
        hip_status = hipMemcpy(load_cb_data_dev.data(),
                               &load_cb_data_host,
                               sizeof(callback_test_data),
                               hipMemcpyHostToDevice);
        if(hip_status != hipSuccess)
        {
            throw hip_runtime_error(
                "Error occurred when copying device memory for loading callback", hip_status);
        }
        load_cb_func.push_back(load_cb_host);
        load_cb_data.push_back(load_cb_data_dev.data());
    };

    if(params.ifields.empty())
    {
        // for library-decomposed multi-GPU, one cb for each device
        if(params.multiGPU > 1)
        {
            for(int i = 0; i < static_cast<int>(params.multiGPU); ++i)
            {
                rocfft_scoped_device dev(i);
                add_load_cb();
            }
        }
        else
        {
            // load cb for current HIP device
            add_load_cb();
        }
    }
    else
    {
        // user-specified decomposition - copy func+data for each brick
        // on this rank
        for(size_t i = 0; i < params.ifields.front().bricks.size(); ++i)
        {
            if(params.ifields.front().bricks[i].rank != mpi_rank)
                continue;

            // load cb for this brick's device
            rocfft_scoped_device dev(params.ifields.front().bricks[i].device);
            add_load_cb();
        }
    }
}

std::shared_ptr<fft_params::jit_cb_state_t> get_rank_jit_state(const fft_params& params,
                                                               const char*       symbol,
                                                               bool              round_trip_inverse,
                                                               jit_callback_op   type)
{
    auto state    = std::make_shared<fft_params::jit_cb_state_t>();
    state->symbol = symbol;

    int mpi_rank = params.get_process_rank();

    switch(type)
    {
    case jit_callback_op::LOAD:
        state->func = get_load_callback_jit(params.itype, params.precision, round_trip_inverse);
        break;
    case jit_callback_op::STORE:
        state->func = get_store_callback_jit(params.otype, params.precision, round_trip_inverse);
        break;
    }

    state->data.resize(rocfft_scoped_device::device_count());

    // If specified does not already have a cbdata allocated for it,
    // alloc callback data pointer and set in the output vec,
    // assuming it's big enough for all devices
    auto add_cb_data_for_device = [&](int deviceID) {
        if(state->data[deviceID])
            return;

        rocfft_scoped_device dev(deviceID);
        callback_test_data   cb_data_host;

        switch(type)
        {
        case jit_callback_op::LOAD:
            if(round_trip_inverse)
            {
                cb_data_host.scalar = params.store_cb_scalar;
            }
            else
            {
                cb_data_host.scalar = params.load_cb_scalar;
            }
            break;
        case jit_callback_op::STORE:
            if(round_trip_inverse)
            {
                cb_data_host.scalar = params.load_cb_scalar;
            }
            else
            {
                cb_data_host.scalar = params.store_cb_scalar;
            }
            break;
        }

        auto hip_status = state->data[deviceID].alloc(sizeof(callback_test_data));
        if(hip_status != hipSuccess)
        {
            throw hip_runtime_error("Error occurred when allocating device memory for callback",
                                    hip_status);
        }
        hip_status = hipMemcpy(state->data[deviceID].data(),
                               &cb_data_host,
                               sizeof(callback_test_data),
                               hipMemcpyHostToDevice);
        if(hip_status != hipSuccess)
        {
            throw hip_runtime_error("Error occurred when copying device memory for callback",
                                    hip_status);
        }
    };

    // user-specified decomposition - alloc data for each brick
    // in the fields on this rank
    auto add_cb_data_for_fields = [&](const std::vector<fft_params::fft_field>& fields) {
        for(const auto& f : fields)
        {
            for(const auto& b : f.bricks)
            {
                if(b.rank != mpi_rank)
                    continue;
                add_cb_data_for_device(b.device);
            }
        }
    };

    if(params.multiGPU > 1)
    {
        // library-decomposed multi-GPU, allocate one cbdata for each device
        for(int device = 0; device < rocfft_scoped_device::device_count(); ++device)
        {
            add_cb_data_for_device(device);
        }
    }
    else
    {
        // check i/o fields
        switch(type)
        {
        case jit_callback_op::LOAD:
            add_cb_data_for_fields(params.ifields);
            break;
        case jit_callback_op::STORE:
            add_cb_data_for_fields(params.ofields);
            break;
        }
        // add cbdata for current device
        add_cb_data_for_device(rocfft_scoped_device::current_device());
    }
    return state;
}

// For the current rank, get a vector of store callback function +
// data pointers.  The pointers need to be in the order that
// fields+bricks were specified to the FFT plan.  Pointers need to be
// copied to the host from the device specified by the respective
// brick.
void get_rank_store_callbacks_funcptr(const fft_params&                          params,
                                      std::vector<void*>&                        store_cb_func,
                                      std::vector<void*>&                        store_cb_data,
                                      bool                                       round_trip_inverse,
                                      std::vector<gpubuf_t<callback_test_data>>& all_cb_data)
{
    int mpi_rank = params.get_process_rank();

    // Copy callback pointer from current device and add to output vec
    auto add_store_cb = [&]() {
        void* store_cb_host
            = get_store_callback_funcptr(params.otype, params.precision, round_trip_inverse);

        callback_test_data store_cb_data_host;

        if(round_trip_inverse)
        {
            store_cb_data_host.scalar = params.load_cb_scalar;
        }
        else
        {
            store_cb_data_host.scalar = params.store_cb_scalar;
        }

        auto& store_cb_data_dev = all_cb_data.emplace_back();
        auto  hip_status        = store_cb_data_dev.alloc(sizeof(callback_test_data));
        if(hip_status != hipSuccess)
        {
            throw hip_runtime_error(
                "Error occurred when allocating device memory for storing callback", hip_status);
        }

        hip_status = hipMemcpy(store_cb_data_dev.data(),
                               &store_cb_data_host,
                               sizeof(callback_test_data),
                               hipMemcpyHostToDevice);
        if(hip_status != hipSuccess)
        {
            throw hip_runtime_error(
                "Error occurred when copying device memory for storing callback", hip_status);
        }

        store_cb_func.push_back(store_cb_host);
        store_cb_data.push_back(store_cb_data_dev.data());
    };

    if(params.ofields.empty())
    {
        // for library-decomposed multi-GPU, one cb for each device
        if(params.multiGPU > 1)
        {
            for(int i = 0; i < static_cast<int>(params.multiGPU); ++i)
            {
                rocfft_scoped_device dev(i);
                add_store_cb();
            }
        }
        else
        {
            // store cb for current HIP device
            add_store_cb();
        }
    }
    else
    {
        // user-specified decomposition - copy func+data for each brick
        // on this rank
        for(size_t i = 0; i < params.ofields.front().bricks.size(); ++i)
        {
            if(params.ofields.front().bricks[i].rank != mpi_rank)
                continue;

            // store cb for this brick's device
            rocfft_scoped_device dev(params.ofields.front().bricks[i].device);
            add_store_cb();
        }
    }
}
