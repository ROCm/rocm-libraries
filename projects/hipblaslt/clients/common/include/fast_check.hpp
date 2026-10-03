// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// fast_check: verification of integer_exact GEMM results without a full host reference GEMM.
//
// A full host reference costs O(M*N*K). fast_check instead uses Freivalds' algorithm: multiply
// the GPU result D by a random vector r and compare against A*(B*r) + C*r + bias*sum(r). The
// expected side costs O(M*K + K*N + M*N) on the host, once per set of inputs; the D side runs on
// the GPU for each solution. Every input is a small integer under integer_exact initialization,
// so the comparison is exact. The sums are taken modulo the prime 2^61-1 with r drawn uniformly
// from [1, 2^61-1), so a wrong D passes a probe with probability below 2^-60. One probe
// multiplies D on the right (finds bad rows) and one on the left (finds bad columns); the two
// lists locate individual bad elements for the failure message.
//
// fast_check also looks outside the M x N result, on the device so that the host never holds
// the padding. The padding of A, B and C is filled with a value outside the integer_exact data
// range, so a kernel that reads padding produces a wrong D. D is filled with a sentinel byte
// pattern before each launch, and afterwards every element of D's allocation outside the
// M x N x batch region must still hold it.

#pragma once

#include <hipblaslt/hipblaslt.h>

#include <cstdint>
#include <string>
#include <vector>

// The bit pattern written over every element of D before each launch, chosen per type, in the
// low bytes. Floating types get a NaN: all-ones bytes, or 0x80 for the FNUZ fp8 types, whose only
// NaN is 0x80. An element that is never written then stands out as non-finite. Integer types have
// no NaN and get their minimum value. fast_check recomputes every integer element that holds the
// type's minimum or maximum exactly, so an unwritten element is still caught, unless its correct
// value saturates to the minimum.
uint64_t fast_check_sentinel_bits(hipDataType type);

// Value written into the padding of A, B and C. integer_exact data lies in [-2, 2]; 40 is far
// outside that range and is exactly representable in every input type fast_check accepts,
// including fp8 e4m3 and e5m2. It is finite so that kernels which load padding and multiply it by
// zero, which is legitimate, still produce 0 rather than NaN.
constexpr float kFastCheckPoisonValue = 40.0f;

// The modulus for the probe sums, the Mersenne prime 2^61 - 1.
constexpr uint64_t kFastCheckModulus = (uint64_t(1) << 61) - 1;

// One column-major matrix in host memory as stored, before any transpose is applied.
struct FastCheckMatrix
{
    const void* data   = nullptr; // first element of batch 0
    hipDataType type   = HIP_R_32F;
    int64_t     rows   = 0; // stored rows
    int64_t     cols   = 0; // stored columns
    int64_t     ld     = 0; // leading dimension, in elements
    int64_t     stride = 0; // distance between batches, in elements
};

// D = scale_d * (alpha * diag(scale_alpha_vec .* scale_a) * op(A) * op(B) * diag(scale_b)
//                + beta * scale_c * C + bias * 1^T), per batch. scale_a is one value or one per
// row of D, and scale_b one value or one per column. Every scale must be an integer.
struct FastCheckProblem
{
    int64_t M           = 0;
    int64_t N           = 0;
    int64_t K           = 0;
    int64_t batch_count = 1;
    bool    transA      = false;
    bool    transB      = false;

    FastCheckMatrix A;
    FastCheckMatrix B;
    FastCheckMatrix C;
    FastCheckMatrix D;

    // Type the GPU accumulates in; bounds the integers it holds exactly.
    hipDataType compute_type = HIP_R_32F;

    double alpha = 1;
    double beta  = 0;

    const void* scale_alpha_vec      = nullptr; // length M, or nullptr
    hipDataType scale_alpha_vec_type = HIP_R_32F;

    const void* bias        = nullptr; // length M per batch, or nullptr
    hipDataType bias_type   = HIP_R_32F;
    int64_t     bias_stride = 0; // distance between per-batch bias vectors, in elements

    const void* scale_a        = nullptr; // one value, or length M when scale_a_vector
    bool        scale_a_vector = false;
    const void* scale_b        = nullptr; // one value, or length N when scale_b_vector
    bool        scale_b_vector = false;
    hipDataType scale_ab_type  = HIP_R_32F;
    double      scale_c        = 1;
    double      scale_d        = 1;

    uint64_t seed = 0; // selects the random probe vectors

    // D's layout on the device, used only to report element offsets in failure messages.
    // Zero means use D's own ld and stride.
    int64_t device_ldd      = 0;
    int64_t device_stride_d = 0;
};

struct FastCheckResult
{
    bool        passed = true;
    std::string message; // empty when passed
};

// Returns true when fast_check can load and round values of this type, and the reason when not.
bool fast_check_supported_type(hipDataType type, std::string* why = nullptr);

// The parts of the check that depend only on the inputs: probe vectors and the expected probe
// sums of D. Compute once and pass to fast_check_result for every solution run on the same
// inputs. problem.D supplies only its type here; its data is not read.
//
// It also bounds, from the actual inputs, the largest magnitude the GPU's accumulator can hold
// at any point: max_partial bounds any partial sum over any subset of K, the form split-K,
// Stream-K and every summation order take, and max_result bounds every finished element of D
// before rounding to D's type. When either reaches the range the compute type holds exactly,
// the result depends on summation order and status fails before any kernel runs.
struct FastCheckExpected
{
    FastCheckResult status; // fails on unsupported, non-integer or inexact inputs
    double          max_partial = 0; // bounds |partial sum|, scaled by alpha and scale or not
    double          max_result  = 0; // bounds |element of D| before rounding to D's type
    std::vector<uint64_t>              row_probe; // r, length N
    std::vector<uint64_t>              col_probe; // t, length M
    std::vector<int64_t>               scale; // scaleAlpha_vector .* scale_a, or ones; length M
    std::vector<int64_t>               col_scale; // scale_b, or ones; length N
    int64_t                            alpha = 0; // alpha * scale_d
    int64_t                            beta  = 0; // beta * scale_c * scale_d
    std::vector<std::vector<int64_t>>  bias; // per batch, length M, times scale_d
    std::vector<std::vector<uint64_t>> row_sums; // per batch, expected D * r
    std::vector<std::vector<uint64_t>> col_sums; // per batch, expected t^T * D
};

FastCheckExpected fast_check_expected(const FastCheckProblem& problem);

// Verifies every element of D inside the M x N x batch region against the expected sums.
FastCheckResult fast_check_result(const FastCheckProblem&  problem,
                                  const FastCheckExpected& expected);

// fast_check_expected followed by fast_check_result.
FastCheckResult fast_check_gemm(const FastCheckProblem& problem);

// fast_check_result with D left in device memory: problem.D describes D's device layout
// (leading dimension and batch stride). The probe sums of D are computed on the GPU, so D is
// never copied to the host. Elements outside the range D's type stores exactly are copied back
// one by one and compared with the exact result. fp8 outputs, and results with more such
// elements than fit in the device list, fall back to copying D and running fast_check_result.
FastCheckResult fast_check_result_device(const FastCheckProblem&  problem,
                                         const FastCheckExpected& expected,
                                         hipStream_t              stream);

// The probe vector entry for the given seed and index, in [1, kFastCheckModulus).
uint64_t fast_check_probe(uint64_t seed, uint64_t index);

// Reads element i of a host buffer of the given type as a double.
double fast_check_load(const void* data, hipDataType type, size_t i);

// Activations fast_check checks exactly. GELU and SiLU have no exact integer form.
enum class FastCheckActivation
{
    none,
    relu, // max(x, 0)
    clamp, // max(arg1, min(x, arg2))
};

// With an activation, D is not linear in the inputs, so it is checked through E, the
// pre-activation result the kernel also writes: E is verified like D (with scale_e in place of
// scale_d), and then every element of D must equal scale_d * act(E / scale_e), rounded to D's
// type. Copies the M x N x batch regions of D and E to the host. Sets *amax, when given, to the
// largest |act(E / scale_e)|, the value amaxD must hold. Fails, and sets *amax to NaN, when an
// element of E is outside the range its type stores exactly.
FastCheckResult fast_check_activation_device(const FastCheckMatrix& d,
                                             const FastCheckMatrix& e,
                                             int64_t                batch_count,
                                             double                 scale_d,
                                             double                 scale_e,
                                             FastCheckActivation    act,
                                             double                 arg1,
                                             double                 arg2,
                                             hipStream_t            stream,
                                             double*                amax);

// The largest |element| of a verified device D divided by scale_d, the value amaxD must hold
// without an activation, or NaN when D holds values outside the range its type stores exactly
// or cannot be copied.
// Copies D's M x N x batch region to the host.
double fast_check_amax_device(const FastCheckMatrix& d,
                              int64_t                batch_count,
                              double                 scale_d,
                              hipStream_t            stream);

// Checks the bias gradient a GEMM writes for one batch, exactly from the inputs: source 'a'
// (BGRADA) sums each row of op(A) over K, and 'b' (BGRADB) each column of op(B). bias holds the
// kernel's output on the host, in bias_type.
FastCheckResult fast_check_bias_gradient(const FastCheckProblem& problem,
                                         char                    source,
                                         const void*             bias,
                                         hipDataType             bias_type);

// Pass/fail for each solution and iteration of a test, so the end of the test can say which
// solutions failed and on which iterations. Each defect class lives in a few of the hundreds of
// shipped solutions, and intermittent defects fail only some iterations, so both are recorded.
class FastCheckSolutionLog
{
public:
    // Records one launch and check. solution is the position in the test's list, library_index
    // the solution's index in the library, and kernel its kernel name, which may be left empty
    // on passing iterations; the last non-empty name is kept.
    void record(
        size_t solution, int library_index, const std::string& kernel, int iteration, bool passed);

    size_t solutions() const
    {
        return entries.size();
    }
    size_t failed_solutions() const;

    // One line per failing solution, naming the solution, its library index and kernel, and the
    // iterations that failed. Empty when every solution passed.
    std::string summary(int iterations) const;

private:
    struct Entry
    {
        size_t           solution;
        int              library_index;
        std::string      kernel;
        std::vector<int> failed_iterations;
    };
    std::vector<Entry> entries;
};

// Changes element (row, col) of the given batch of a device matrix to a different value: the
// sentinel when the element does not already hold it, and the poison value otherwise. Used by
// the fast_check_inject self-test to prove that a corrupted result is reported on the iteration
// that produced it.
hipError_t fast_check_corrupt_element_device(
    const FastCheckMatrix& m, int64_t batch, int64_t row, int64_t col, hipStream_t stream);

// One device buffer a GEMM reads or writes, for the placement record in failure messages.
struct FastCheckBuffer
{
    const char* name;
    const void* base;
    size_t      bytes;
};

// Lists each buffer's address range, one per line, and flags any range that crosses a 4 GiB
// boundary. A kernel that drops the carry out of the low 32 bits of an address goes wrong only
// past such a boundary, so the record shows whether a failure lines up with one.
std::string fast_check_describe_buffers(const std::vector<FastCheckBuffer>& buffers);

// Device helpers. A device matrix is described by FastCheckMatrix with data pointing to device
// memory; total_elements is the size of its allocation in elements, which may extend past the
// last batch.

// Bytes per element, or 0 when fast_check does not support the type.
size_t fast_check_element_size(hipDataType type);

// The bit pattern of kFastCheckPoisonValue in the given type, in the low bytes.
uint64_t fast_check_poison_bits(hipDataType type);

// Counts the elements of a device range that no longer hold kFastCheckPoisonValue.
struct FastCheckChanged
{
    uint64_t count = 0;
    uint64_t first = 0; // index of the first changed element, when count > 0
    bool     ok    = true; // false when the scan itself failed
};
FastCheckChanged fast_check_count_changed_device(const void* data,
                                                 hipDataType type,
                                                 size_t      elements,
                                                 hipStream_t stream);

// Fills every element of a device buffer with fast_check_sentinel_bits(type). Returns the launch
// error, or hipErrorInvalidValue for a type fast_check does not support.
hipError_t fast_check_fill_sentinel_device(void*       buffer,
                                           hipDataType type,
                                           size_t      elements,
                                           hipStream_t stream);

// Writes kFastCheckPoisonValue into every element outside the rows x cols x batch region. Returns
// as fast_check_fill_sentinel_device does.
hipError_t fast_check_poison_padding_device(const FastCheckMatrix& m,
                                            int64_t                batch_count,
                                            size_t                 total_elements,
                                            hipStream_t            stream);

// Checks every element outside the rows x cols x batch region. With expect_poison, each must hold
// kFastCheckPoisonValue (used when D shares C's poisoned memory); otherwise each must hold the
// sentinel pattern. Without expect_poison, and when the sentinel decodes to NaN in the matrix
// type, an element inside the region that still holds the sentinel is reported as never written.
FastCheckResult fast_check_scan_padding_device(const FastCheckMatrix& m,
                                               int64_t                batch_count,
                                               size_t                 total_elements,
                                               bool                   expect_poison,
                                               hipStream_t            stream);

// Copies the rows x cols x batch region of a device matrix into contiguous host memory
// (leading dimension rows, batch stride rows * cols). host must hold that many elements.
hipError_t fast_check_copy_region_to_host(void*                  host,
                                          const FastCheckMatrix& device_matrix,
                                          int64_t                batch_count,
                                          hipStream_t            stream);
