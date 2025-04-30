#include "rtc_partial_pass_sbrr_64_64_64.h"
#include "device/kernel-generator-embed.h"
#include "include/kernel_launch.h"
#include "rtc_chirp_kernel.h"
#include "rtc_kernel.h"
#include "tree_node.h"

std::string partial_pass_64_64_64_sbrr_rtc_kernel_name(rocfft_precision        precision,
                                                       int                     direction,
                                                       rocfft_result_placement placement,
                                                       rocfft_array_type       inArrayType,
                                                       rocfft_array_type       outArrayType,
                                                       CallbackType            cbtype)
{
    std::string kernel_name = "sbrr_64_64_64_partial_pass";

    if(direction == -1)
        kernel_name += "_fwd";
    else
        kernel_name += "_bck";

    if(placement == rocfft_placement_inplace)
    {
        kernel_name += "_ip";
        kernel_name += rtc_array_type_name(inArrayType);
    }
    else
    {
        kernel_name += "_op";
        kernel_name += rtc_array_type_name(inArrayType);
        kernel_name += rtc_array_type_name(outArrayType);
    }

    kernel_name += rtc_precision_name(precision);

    kernel_name += rtc_cbtype_name(cbtype);

    return kernel_name;
}

static std::string lds_to_reg()
{
    std::string body = R"_SRC(
        template <typename scalar_type, StrideBin sb>
__device__ void lds_to_reg_input_length64_device_sbrr(scalar_type *R,
                                                      scalar_type *__restrict__ lds_complex,
                                                      unsigned int stride_lds,
                                                      unsigned int offset_lds,
                                                      unsigned int thread,
                                                      bool write)
{
       const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
       unsigned int l_offset;
       __syncthreads();
       l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
       R[0] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 8) * lstride;
       R[1] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 16) * lstride;
       R[2] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 24) * lstride;
       R[3] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 32) * lstride;
       R[4] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 40) * lstride;
       R[5] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 48) * lstride;
       R[6] = lds_complex[l_offset];
       l_offset = offset_lds + ((thread + 0 + 0) + 56) * lstride;
       R[7] = lds_complex[l_offset];
}
)_SRC";

    return body;
}

static std::string reg_to_lds()
{
    std::string body = R"_SRC(
template <typename scalar_type, StrideBin sb>
__device__ void lds_from_reg_output_length64_device_sbrr(scalar_type *R,
                                                         scalar_type *__restrict__ lds_complex,
                                                         unsigned int stride_lds,
                                                         unsigned int offset_lds,
                                                         unsigned int thread,
                                                         bool write)
{
       const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
       unsigned int l_offset;
       __syncthreads();
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 0) * lstride;
       lds_complex[l_offset] = R[0];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 8) * lstride;
       lds_complex[l_offset] = R[1];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 16) * lstride;
       lds_complex[l_offset] = R[2];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 24) * lstride;
       lds_complex[l_offset] = R[3];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 32) * lstride;
       lds_complex[l_offset] = R[4];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 40) * lstride;
       lds_complex[l_offset] = R[5];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 48) * lstride;
       lds_complex[l_offset] = R[6];
       l_offset = offset_lds + (((thread + 0 + 0) / 8) * 64 + (thread + 0 + 0) % 8 + 56) * lstride;
       lds_complex[l_offset] = R[7];
}
)_SRC";

    return body;
}

static std::string lds_to_reg_pp()
{
    std::string body = R"_SRC(
    template <typename scalar_type>
__device__ void lds_to_reg_16_input_length64_device_pp(scalar_type *R,
                                                       scalar_type *__restrict__ lds_complex,
                                                       unsigned int stride,
                                                       unsigned int offset)
{
       unsigned int idx, thread;
       __syncthreads();

       thread = 0;
       idx = offset + thread * stride;
       R[0] = lds_complex[idx];

       thread = 1;
       idx = offset + thread * stride;
       R[1] = lds_complex[idx];

       thread = 2;
       idx = offset + thread * stride;
       R[2] = lds_complex[idx];

       thread = 3;
       idx = offset + thread * stride;
       R[3] = lds_complex[idx];

       thread = 4;
       idx = offset + thread * stride;
       R[4] = lds_complex[idx];

       thread = 5;
       idx = offset + thread * stride;
       R[5] = lds_complex[idx];

       thread = 6;
       idx = offset + thread * stride;
       R[6] = lds_complex[idx];

       thread = 7;
       idx = offset + thread * stride;
       R[7] = lds_complex[idx];

       thread = 8;
       idx = offset + thread * stride;
       R[8] = lds_complex[idx];

       thread = 9;
       idx = offset + thread * stride;
       R[9] = lds_complex[idx];

       thread = 10;
       idx = offset + thread * stride;
       R[10] = lds_complex[idx];

       thread = 11;
       idx = offset + thread * stride;
       R[11] = lds_complex[idx];

       thread = 12;
       idx = offset + thread * stride;
       R[12] = lds_complex[idx];

       thread = 13;
       idx = offset + thread * stride;
       R[13] = lds_complex[idx];

       thread = 14;
       idx = offset + thread * stride;
       R[14] = lds_complex[idx];

       thread = 15;
       idx = offset + thread * stride;
       R[15] = lds_complex[idx];
}
)_SRC";

    return body;
}

static std::string reg_to_lds_pp()
{
    std::string body = R"_SRC(
    template <typename scalar_type>
__device__ void lds_from_reg_16_output_length64_device_pp(scalar_type *R,
                                                          scalar_type *__restrict__ lds_complex,
                                                          unsigned int stride,
                                                          unsigned int offset)
{
       unsigned int idx, thread;
       __syncthreads();

       thread = 0;
       idx = offset + thread * stride;
       lds_complex[idx] = R[0];

       thread = 1;
       idx = offset + thread * stride;
       lds_complex[idx] = R[1];

       thread = 2;
       idx = offset + thread * stride;
       lds_complex[idx] = R[2];

       thread = 3;
       idx = offset + thread * stride;
       lds_complex[idx] = R[3];

       thread = 4;
       idx = offset + thread * stride;
       lds_complex[idx] = R[4];

       thread = 5;
       idx = offset + thread * stride;
       lds_complex[idx] = R[5];

       thread = 6;
       idx = offset + thread * stride;
       lds_complex[idx] = R[6];

       thread = 7;
       idx = offset + thread * stride;
       lds_complex[idx] = R[7];

       thread = 8;
       idx = offset + thread * stride;
       lds_complex[idx] = R[8];

       thread = 9;
       idx = offset + thread * stride;
       lds_complex[idx] = R[9];

       thread = 10;
       idx = offset + thread * stride;
       lds_complex[idx] = R[10];

       thread = 11;
       idx = offset + thread * stride;
       lds_complex[idx] = R[11];

       thread = 12;
       idx = offset + thread * stride;
       lds_complex[idx] = R[12];

       thread = 13;
       idx = offset + thread * stride;
       lds_complex[idx] = R[13];

       thread = 14;
       idx = offset + thread * stride;
       lds_complex[idx] = R[14];

       thread = 15;
       idx = offset + thread * stride;
       lds_complex[idx] = R[15];
}
)_SRC";

    return body;
}

static std::string twiddle_multiply_pp_fwd()
{
    std::string body = R"_SRC(
    template <typename scalar_type>
__device__ void twiddle_multiple_pp_fwd(scalar_type *R,
                                        unsigned int thread,
                                        const scalar_type *__restrict__ twiddles_pp)
{
       scalar_type t;
       scalar_type W;

       W = twiddles_pp[thread * 64 + 0];
       t = {R[0].x * W.x - R[0].y * W.y, R[0].y * W.x + R[0].x * W.y};
       R[0] = t;

       W = twiddles_pp[thread * 64 + 1];
       t = {R[1].x * W.x - R[1].y * W.y, R[1].y * W.x + R[1].x * W.y};
       R[1] = t;
       
       W = twiddles_pp[thread * 64 + 2];
       t = {R[2].x * W.x - R[2].y * W.y, R[2].y * W.x + R[2].x * W.y};
       R[2] = t;
       
       W = twiddles_pp[thread * 64 + 3];
       t = {R[3].x * W.x - R[3].y * W.y, R[3].y * W.x + R[3].x * W.y};
       R[3] = t;
       
       W = twiddles_pp[thread * 64 + 4];
       t = {R[4].x * W.x - R[4].y * W.y, R[4].y * W.x + R[4].x * W.y};
       R[4] = t;
       
       W = twiddles_pp[thread * 64 + 5];
       t = {R[5].x * W.x - R[5].y * W.y, R[5].y * W.x + R[5].x * W.y};
       R[5] = t;
       
       W = twiddles_pp[thread * 64 + 6];
       t = {R[6].x * W.x - R[6].y * W.y, R[6].y * W.x + R[6].x * W.y};
       R[6] = t;
       
       W = twiddles_pp[thread * 64 + 7];
       t = {R[7].x * W.x - R[7].y * W.y, R[7].y * W.x + R[7].x * W.y};
       R[7] = t;
       
       W = twiddles_pp[thread * 64 + 8];
       t = {R[8].x * W.x - R[8].y * W.y, R[8].y * W.x + R[8].x * W.y};
       R[8] = t;
       
       W = twiddles_pp[thread * 64 + 9];
       t = {R[9].x * W.x - R[9].y * W.y, R[9].y * W.x + R[9].x * W.y};
       R[9] = t;
       
       W = twiddles_pp[thread * 64 + 10];
       t = {R[10].x * W.x - R[10].y * W.y, R[10].y * W.x + R[10].x * W.y};
       R[10] = t;
       
       W = twiddles_pp[thread * 64 + 11];
       t = {R[11].x * W.x - R[11].y * W.y, R[11].y * W.x + R[11].x * W.y};
       R[11] = t;
       
       W = twiddles_pp[thread * 64 + 12];
       t = {R[12].x * W.x - R[12].y * W.y, R[12].y * W.x + R[12].x * W.y};
       R[12] = t;
       
       W = twiddles_pp[thread * 64 + 13];
       t = {R[13].x * W.x - R[13].y * W.y, R[13].y * W.x + R[13].x * W.y};
       R[13] = t;
       
       W = twiddles_pp[thread * 64 + 14];
       t = {R[14].x * W.x - R[14].y * W.y, R[14].y * W.x + R[14].x * W.y};
       R[14] = t;
       
       W = twiddles_pp[thread * 64 + 15];
       t = {R[15].x * W.x - R[15].y * W.y, R[15].y * W.x + R[15].x * W.y};
       R[15] = t;
}
)_SRC";

    return body;
}

static std::string twiddle_multiply_pp_inv()
{
    std::string body = R"_SRC(
    template <typename scalar_type>
__device__ void twiddle_multiple_pp_inv(scalar_type *R,
                                        unsigned int thread,
                                        const scalar_type *__restrict__ twiddles_pp)
{
       scalar_type t;
       scalar_type W;

       W = twiddles_pp[thread * 64 + 0];
       t = {R[0].x * W.x + R[0].y * W.y, R[0].y * W.x - R[0].x * W.y};
       R[0] = t;

       W = twiddles_pp[thread * 64 + 1];
       t = {R[1].x * W.x + R[1].y * W.y, R[1].y * W.x - R[1].x * W.y};
       R[1] = t;
       
       W = twiddles_pp[thread * 64 + 2];
       t = {R[2].x * W.x + R[2].y * W.y, R[2].y * W.x - R[2].x * W.y};
       R[2] = t;
       
       W = twiddles_pp[thread * 64 + 3];
       t = {R[3].x * W.x + R[3].y * W.y, R[3].y * W.x - R[3].x * W.y};
       R[3] = t;
       
       W = twiddles_pp[thread * 64 + 4];
       t = {R[4].x * W.x + R[4].y * W.y, R[4].y * W.x - R[4].x * W.y};
       R[4] = t;
       
       W = twiddles_pp[thread * 64 + 5];
       t = {R[5].x * W.x + R[5].y * W.y, R[5].y * W.x - R[5].x * W.y};
       R[5] = t;
       
       W = twiddles_pp[thread * 64 + 6];
       t = {R[6].x * W.x + R[6].y * W.y, R[6].y * W.x - R[6].x * W.y};
       R[6] = t;
       
       W = twiddles_pp[thread * 64 + 7];
       t = {R[7].x * W.x + R[7].y * W.y, R[7].y * W.x - R[7].x * W.y};
       R[7] = t;
       
       W = twiddles_pp[thread * 64 + 8];
       t = {R[8].x * W.x + R[8].y * W.y, R[8].y * W.x - R[8].x * W.y};
       R[8] = t;
       
       W = twiddles_pp[thread * 64 + 9];
       t = {R[9].x * W.x + R[9].y * W.y, R[9].y * W.x - R[9].x * W.y};
       R[9] = t;
       
       W = twiddles_pp[thread * 64 + 10];
       t = {R[10].x * W.x + R[10].y * W.y, R[10].y * W.x - R[10].x * W.y};
       R[10] = t;
       
       W = twiddles_pp[thread * 64 + 11];
       t = {R[11].x * W.x + R[11].y * W.y, R[11].y * W.x - R[11].x * W.y};
       R[11] = t;
       
       W = twiddles_pp[thread * 64 + 12];
       t = {R[12].x * W.x + R[12].y * W.y, R[12].y * W.x - R[12].x * W.y};
       R[12] = t;
       
       W = twiddles_pp[thread * 64 + 13];
       t = {R[13].x * W.x + R[13].y * W.y, R[13].y * W.x - R[13].x * W.y};
       R[13] = t;
       
       W = twiddles_pp[thread * 64 + 14];
       t = {R[14].x * W.x + R[14].y * W.y, R[14].y * W.x - R[14].x * W.y};
       R[14] = t;
       
       W = twiddles_pp[thread * 64 + 15];
       t = {R[15].x * W.x + R[15].y * W.y, R[15].y * W.x - R[15].x * W.y};
       R[15] = t;
}
)_SRC";

    return body;
}

static std::string fwd_len64()
{
    std::string body = R"_SRC(
    template <typename scalar_type,
          const bool lds_is_real,
          StrideBin sb,
          const bool lds_linear,
          const bool direct_load_to_reg>
__device__ void forward_length64_SBRR_device_pp(scalar_type *R,
                                                real_type_t<scalar_type> *__restrict__ lds_real,
                                                scalar_type *__restrict__ lds_complex,
                                                const scalar_type *__restrict__ twiddles,
                                                unsigned int stride_lds,
                                                unsigned int offset_lds,
                                                unsigned int thread,
                                                bool write)
{
    scalar_type W;
    scalar_type t;
    const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
    unsigned int l_offset;

    // pass 0, width 8
    // using 8 threads we need to do 8 radix-8 butterflies
    // therefore each thread will do 1.000000 butterflies
    FwdRad8B1(R + 0, R + 1, R + 2, R + 3, R + 4, R + 5, R + 6, R + 7);

    __syncthreads();
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 0) * lstride;
    lds_complex[l_offset] = R[0];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 1) * lstride;
    lds_complex[l_offset] = R[1];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 2) * lstride;
    lds_complex[l_offset] = R[2];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 3) * lstride;
    lds_complex[l_offset] = R[3];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 4) * lstride;
    lds_complex[l_offset] = R[4];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 5) * lstride;
    lds_complex[l_offset] = R[5];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 6) * lstride;
    lds_complex[l_offset] = R[6];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 7) * lstride;
    lds_complex[l_offset] = R[7];

    // pass 1, width 8
    // using 8 threads we need to do 8 radix-8 butterflies
    // therefore each thread will do 1.000000 butterflies
    __syncthreads();
    l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
    R[0] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 8) * lstride;
    R[1] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 16) * lstride;
    R[2] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 24) * lstride;
    R[3] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 32) * lstride;
    R[4] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 40) * lstride;
    R[5] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 48) * lstride;
    R[6] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 56) * lstride;
    R[7] = lds_complex[l_offset];

    W = twiddles[0 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[1].x * W.x - R[1].y * W.y, R[1].y * W.x + R[1].x * W.y};
    R[1] = t;
    W = twiddles[1 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[2].x * W.x - R[2].y * W.y, R[2].y * W.x + R[2].x * W.y};
    R[2] = t;
    W = twiddles[2 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[3].x * W.x - R[3].y * W.y, R[3].y * W.x + R[3].x * W.y};
    R[3] = t;
    W = twiddles[3 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[4].x * W.x - R[4].y * W.y, R[4].y * W.x + R[4].x * W.y};
    R[4] = t;
    W = twiddles[4 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[5].x * W.x - R[5].y * W.y, R[5].y * W.x + R[5].x * W.y};
    R[5] = t;
    W = twiddles[5 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[6].x * W.x - R[6].y * W.y, R[6].y * W.x + R[6].x * W.y};
    R[6] = t;
    W = twiddles[6 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[7].x * W.x - R[7].y * W.y, R[7].y * W.x + R[7].x * W.y};
    R[7] = t;
    FwdRad8B1(R + 0, R + 1, R + 2, R + 3, R + 4, R + 5, R + 6, R + 7);
}
)_SRC";

    return body;
}

static std::string inv_len64()
{
    std::string body = R"_SRC(
    template <typename scalar_type,
          const bool lds_is_real,
          StrideBin sb,
          const bool lds_linear,
          const bool direct_load_to_reg>
__device__ void inverse_length64_SBRR_device_pp(scalar_type *R,
                                                real_type_t<scalar_type> *__restrict__ lds_real,
                                                scalar_type *__restrict__ lds_complex,
                                                const scalar_type *__restrict__ twiddles,
                                                unsigned int stride_lds,
                                                unsigned int offset_lds,
                                                unsigned int thread,
                                                bool write)
{
    scalar_type W;
    scalar_type t;
    const unsigned int lstride = (sb == SB_UNIT) ? (1) : (stride_lds);
    unsigned int l_offset;

    // pass 0, width 8
    // using 8 threads we need to do 8 radix-8 butterflies
    // therefore each thread will do 1.000000 butterflies
    InvRad8B1(R + 0, R + 1, R + 2, R + 3, R + 4, R + 5, R + 6, R + 7);

    __syncthreads();
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 0) * lstride;
    lds_complex[l_offset] = R[0];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 1) * lstride;
    lds_complex[l_offset] = R[1];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 2) * lstride;
    lds_complex[l_offset] = R[2];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 3) * lstride;
    lds_complex[l_offset] = R[3];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 4) * lstride;
    lds_complex[l_offset] = R[4];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 5) * lstride;
    lds_complex[l_offset] = R[5];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 6) * lstride;
    lds_complex[l_offset] = R[6];
    l_offset = offset_lds + (((thread + 0 + 0) / 1) * 8 + (thread + 0 + 0) % 1 + 7) * lstride;
    lds_complex[l_offset] = R[7];

    // pass 1, width 8
    // using 8 threads we need to do 8 radix-8 butterflies
    // therefore each thread will do 1.000000 butterflies
    __syncthreads();
    l_offset = offset_lds + ((thread + 0 + 0) + 0) * lstride;
    R[0] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 8) * lstride;
    R[1] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 16) * lstride;
    R[2] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 24) * lstride;
    R[3] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 32) * lstride;
    R[4] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 40) * lstride;
    R[5] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 48) * lstride;
    R[6] = lds_complex[l_offset];
    l_offset = offset_lds + ((thread + 0 + 0) + 56) * lstride;
    R[7] = lds_complex[l_offset];

    W = twiddles[0 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[1].x * W.x + R[1].y * W.y, R[1].y * W.x - R[1].x * W.y};
    R[1] = t;
    W = twiddles[1 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[2].x * W.x + R[2].y * W.y, R[2].y * W.x - R[2].x * W.y};
    R[2] = t;
    W = twiddles[2 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[3].x * W.x + R[3].y * W.y, R[3].y * W.x - R[3].x * W.y};
    R[3] = t;
    W = twiddles[3 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[4].x * W.x + R[4].y * W.y, R[4].y * W.x - R[4].x * W.y};
    R[4] = t;
    W = twiddles[4 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[5].x * W.x + R[5].y * W.y, R[5].y * W.x - R[5].x * W.y};
    R[5] = t;
    W = twiddles[5 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[6].x * W.x + R[6].y * W.y, R[6].y * W.x - R[6].x * W.y};
    R[6] = t;
    W = twiddles[6 + 7 * ((thread + 0 + 0) % 8)];
    t = {R[7].x * W.x + R[7].y * W.y, R[7].y * W.x - R[7].x * W.y};
    R[7] = t;
    InvRad8B1(R + 0, R + 1, R + 2, R + 3, R + 4, R + 5, R + 6, R + 7);
}
)_SRC";

    return body;
}

static std::string pp_fwd_step_1_2_radix16()
{
    std::string body = R"_SRC(
    // Partial pass step 1: length-16 DFT on off-dimension

    // Radix-16 pass
    FwdRad16B1(R + 0, R + 1, R + 2, R + 3, R + 4, R + 5, R + 6, R + 7, R + 8, R + 9, R + 10, R + 11, R + 12, R + 13, R + 14, R + 15);

    // Partial pass step 2: Hadamard product with twiddle factors
    twiddle_multiple_pp_fwd<scalar_type>(R, blockIdx.x % 4, twiddles_pp);
)_SRC";

    return body;
}

static std::string pp_inv_step_1_2_radix16()
{
    std::string body = R"_SRC(
    // Partial pass step 1: length-16 DFT on off-dimension

    // Radix-16 pass
    InvRad16B1(R + 0, R + 1, R + 2, R + 3, R + 4, R + 5, R + 6, R + 7, R + 8, R + 9, R + 10, R + 11, R + 12, R + 13, R + 14, R + 15);

    // Partial pass step 2: Hadamard product with twiddle factors
    twiddle_multiple_pp_inv<scalar_type>(R, blockIdx.x % 4, twiddles_pp);
)_SRC";

    return body;
}

// TODO global function name is hardcoded
static std::string partial_pass_sbrr_64_64_64_rtc_body(const std::string& kernel_name,
                                                       int                direction)
{
    std::string body;

    body += R"_SRC(
    extern "C" __global__
    __launch_bounds__(128) void 
    )_SRC";

    body += kernel_name;

    body += R"_SRC(
    (
        const scalar_type *__restrict__ twiddles_pp,
        const scalar_type *__restrict__ twiddles,
        const size_t *__restrict__ lengths,
        const size_t *__restrict__ stride,
        const size_t nbatch,
        void *__restrict__ load_cb_fn,
        void *__restrict__ load_cb_data,
        unsigned int load_cb_lds_bytes,
        void *__restrict__ store_cb_fn,
        void *__restrict__ store_cb_data,
        scalar_type *__restrict__ ibuf,
        scalar_type *__restrict__ obuf)
{
    auto const sb = SB_UNIT;
    auto const ebtype = EmbeddedType::NONE;
    auto const sbrc_type = SBRC_2D;
    auto const transpose_type = NONE;
    auto const drtype = DirectRegType::FORCE_OFF_OR_NOT_SUPPORT;
    auto const apply_large_twiddle = false;
    auto const intrinsic_mode = IntrinsicAccessType::DISABLE_BOTH;
    const size_t large_twiddle_base = 8;
    const size_t large_twiddle_steps = 0;

    // this kernel:
    //   uses 8 threads per transform
    //   does 16 transforms per thread block
    // therefore it should be called with 128 threads per thread block
    scalar_type R[16];
    extern __shared__ unsigned char __attribute__((aligned(sizeof(scalar_type)))) lds_uchar[];
    real_type_t<scalar_type> *__restrict__ lds_real = reinterpret_cast<real_type_t<scalar_type> *>(lds_uchar);
    scalar_type *__restrict__ lds_complex = reinterpret_cast<scalar_type *>(lds_uchar);
    size_t offset = 0;
    unsigned int offset_lds;
    unsigned int stride_lds;
    size_t batch;
    size_t transform;
    const bool direct_load_to_reg = false;
    const bool direct_store_from_reg = false;
    const bool lds_linear = true;
    const bool lds_is_real = false;
    auto load_cb = get_load_cb<scalar_type, cbtype>(load_cb_fn);
    auto store_cb = get_store_cb<scalar_type, cbtype>(store_cb_fn);

    size_t global_mem_idx = 0, offset_pp = 0, remaining_pp = 0;

    const StrideBin SB_1ST = (ebtype == EmbeddedType::C2Real_PRE) ? SB_NONUNIT : SB_UNIT;
    const StrideBin SB_2ND = (ebtype == EmbeddedType::C2Real_PRE) ? SB_UNIT : SB_NONUNIT;

    // large twiddles
    // - no large twiddles

     // offsets
    const size_t dim = 3;
    // const size_t stride0 = (sb == SB_UNIT) ? (1) : (stride[0]);
    const size_t stride0 = (stride[0]);
    unsigned int thread;
    size_t remaining;
    size_t index_along_d;
    transform = blockIdx.x * 16 + threadIdx.x / 8;
    remaining = transform;
    remaining_pp = 64 * (transform / 64) + (transform % 64) / 16 + (transform * 4) % 64;
    for (int d = 1; d < dim; ++d)
    {
        // index_along_d = remaining % lengths[d];
        remaining = remaining / lengths[d];
        // offset = offset + index_along_d * stride[d];

        index_along_d = remaining_pp % lengths[d];
        remaining_pp = remaining_pp / lengths[d];
        offset_pp = offset_pp + index_along_d * stride[d];
    }
    batch = remaining;
    // offset = offset + batch * stride[dim];
    offset_pp = offset_pp + batch * stride[dim];
    stride_lds = 64 + (ebtype == EmbeddedType::NONE ? 0 : 1);
    offset_lds = stride_lds * (transform % 16);   

    bool inbound = batch < nbatch;

    // load global into lds
    if (inbound)
    {
        thread = threadIdx.x % 8;

        global_mem_idx = offset_pp + (thread + 0) * stride0;
        lds_complex[offset_lds + (thread + 0)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 8) * stride0;
        lds_complex[offset_lds + (thread + 8)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 16) * stride0;
        lds_complex[offset_lds + (thread + 16)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 24) * stride0;
        lds_complex[offset_lds + (thread + 24)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 32) * stride0;
        lds_complex[offset_lds + (thread + 32)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 40) * stride0;
        lds_complex[offset_lds + (thread + 40)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 48) * stride0;
        lds_complex[offset_lds + (thread + 48)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);

        global_mem_idx = offset_pp + (thread + 56) * stride0;
        lds_complex[offset_lds + (thread + 56)] = load_cb(ibuf, global_mem_idx, load_cb_data, nullptr);
    }

    // calc the thread_in_device value once and for all device funcs
    unsigned int thread_in_device = lds_linear ? threadIdx.x % 8 : threadIdx.x / 16;

    // call a pre-load from lds to registers (if necessary)
    lds_to_reg_input_length64_device_sbrr<scalar_type, SB_1ST>(
        R, lds_complex, stride_lds, offset_lds, thread_in_device, true); 
    )_SRC";

    if(direction == -1)
    {
        body += R"_SRC(
    // transform
    forward_length64_SBRR_device_pp<scalar_type,
                                    lds_is_real,
                                    SB_1ST,
                                    lds_linear,
                                    direct_load_to_reg>(
        R, lds_real, lds_complex, twiddles, stride_lds, offset_lds, thread_in_device, true);        
    )_SRC";
    }
    else if(direction == 1)
    {
        body += R"_SRC(
    // transform
    inverse_length64_SBRR_device_pp<scalar_type,
                                    lds_is_real,
                                    SB_1ST,
                                    lds_linear,
                                    direct_load_to_reg>(
        R, lds_real, lds_complex, twiddles, stride_lds, offset_lds, thread_in_device, true);        
    )_SRC";
    }

    body += R"_SRC(        
    // call a post-store from registers to lds (if necessary)
    lds_from_reg_output_length64_device_sbrr<scalar_type, SB_1ST>(
        R, lds_complex, stride_lds, offset_lds, thread_in_device, true);

    auto stride_lds_pp = 64;
    auto offset_lds_pp = (blockIdx.x * 16 + threadIdx.x) % 64;

    // call a pre-load from lds to registers (if necessary)
    lds_to_reg_16_input_length64_device_pp<scalar_type>(R, lds_complex, stride_lds_pp, offset_lds_pp);
    )_SRC";

    if(direction == -1)
        body += pp_fwd_step_1_2_radix16();
    else if(direction == 1)
        body += pp_inv_step_1_2_radix16();

    body += R"_SRC(
    // call a post-store from registers to lds (if necessary)
    lds_from_reg_16_output_length64_device_pp<scalar_type>(R, lds_complex, stride_lds_pp, offset_lds_pp);

    // store global
    __syncthreads();
    if (inbound)
    {
        global_mem_idx = offset_pp + (thread + 0) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 0)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 8) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 8)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 16) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 16)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 24) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 24)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 32) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 32)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 40) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 40)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 48) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 48)],
                 store_cb_data,
                 nullptr);

        global_mem_idx = offset_pp + (thread + 56) * stride0;
        store_cb(obuf,
                 global_mem_idx,
                 lds_complex[offset_lds + (thread + 56)],
                 store_cb_data,
                 nullptr);
    }
}
        )_SRC";

    return body;
}

std::string partial_pass_sbrr_64_64_64_rtc(const std::string&              kernel_name,
                                           const std::vector<unsigned int> all_factors,
                                           int                             direction,
                                           rocfft_precision                precision,
                                           CallbackType                    cbtype)
{
    std::string src;

    // start off with includes
    src += rocfft_complex_h;
    src += common_h;
    src += memory_gfx_h;
    src += callback_h;
    src += butterfly_constant_h;

    append_radix_h(src, all_factors);

    src += rtc_precision_type_decl(precision);
    src += lds_to_reg();
    src += reg_to_lds();
    src += lds_to_reg_pp();
    src += reg_to_lds_pp();
    if(direction == -1)
    {
        src += fwd_len64();
        src += twiddle_multiply_pp_fwd();
    }
    else if(direction == 1)
    {
        src += inv_len64();
        src += twiddle_multiply_pp_inv();
    }
    src += rtc_const_cbtype_decl(cbtype);
    src += partial_pass_sbrr_64_64_64_rtc_body(kernel_name, direction);

    return src;
}

RTCKernel::RTCGenerator RTCKernelPartialPassSBRR64Cubed::generate_from_node(
    const LeafNode& node, const std::string& gpu_arch, bool enable_callbacks)
{
    RTCGenerator generator;

    auto scheme = node.scheme;
    if(!(scheme == CS_KERNEL_STOCKHAM && node.applyPartialPass))
        return generator;

    size_t batch_accum = node.batch;
    for(size_t j = 1; j < node.length.size(); j++)
        batch_accum *= node.length[j];

    auto workgroup_size        = PARTIAL_PASS_SBRR_64_64_64_THREADS;
    auto threads_per_transform = 8;
    auto transforms_per_block  = workgroup_size / threads_per_transform;

    auto bwd = transforms_per_block;

    auto b_x   = (batch_accum + bwd - 1) / bwd;
    auto wgs_x = workgroup_size;

    generator.gridDim  = {static_cast<unsigned int>(b_x), 1, 1};
    generator.blockDim = {wgs_x, 1, 1};

    auto precision    = node.precision;
    auto direction    = node.direction;
    auto placement    = node.placement;
    auto inArrayType  = node.inArrayType;
    auto outArrayType = node.outArrayType;
    auto cbType       = node.GetCallbackType(enable_callbacks);

    auto kernelFactors   = node.kernelFactors;
    auto kernelFactorsPP = node.kernelFactorsPP;

    std::vector<unsigned int> all_factors(kernelFactors.begin(), kernelFactors.end());
    all_factors.insert(all_factors.end(), kernelFactorsPP.begin(), kernelFactorsPP.end());

    generator.generate_name = [=]() {
        return partial_pass_64_64_64_sbrr_rtc_kernel_name(
            precision, direction, placement, inArrayType, outArrayType, cbType);
    };

    generator.generate_src = [=](const std::string& kernel_name) {
        return partial_pass_sbrr_64_64_64_rtc(
            kernel_name, all_factors, direction, precision, cbType);
    };

    generator.construct_rtckernel = [=](const std::string&       kernel_name,
                                        const std::vector<char>& code,
                                        dim3                     gridDim,
                                        dim3                     blockDim) {
        return std::unique_ptr<RTCKernel>(
            new RTCKernelPartialPassSBRR64Cubed(kernel_name, code, gridDim, blockDim));
    };

    return generator;
}

RTCKernelArgs RTCKernelPartialPassSBRR64Cubed::get_launch_args(DeviceCallIn& data)
{
    RTCKernelArgs kargs;

    kargs.append_ptr(data.node->twiddles_pp);
    kargs.append_ptr(data.node->twiddles);

    kargs.append_ptr(kargs_lengths(data.node->devKernArg));
    kargs.append_ptr(kargs_stride_in(data.node->devKernArg));
    kargs.append_size_t(data.node->batch);

    // callback params
    kargs.append_ptr(data.callbacks.load_cb_fn);
    kargs.append_ptr(data.callbacks.load_cb_data);
    kargs.append_unsigned_int(data.callbacks.load_cb_lds_bytes);
    kargs.append_ptr(data.callbacks.store_cb_fn);
    kargs.append_ptr(data.callbacks.store_cb_data);
    append_load_store_args(kargs, *data.node);

    kargs.append_ptr(data.bufIn[0]);
    kargs.append_ptr(data.bufOut[0]);

    return kargs;
}
