// MIT License
//
// Copyright (c) 2017-2020 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "experimental/sparse_matrix.hpp"

#include "hipcub/device/device_spmv.hpp"
#include "hipcub/util_allocator.hpp"

#include "common_test_header.hpp"

hipcub::CachingDeviceAllocator g_allocator;

static constexpr float alpha = 1.0f;
static constexpr float beta = 0.0f;

// Params for tests
template<
    class Type,
    int32_t Grid2D = -1,
    int32_t Grid3D = -1,
    int32_t Wheel = -1,
    int32_t Dense = -1
>
struct DeviceSpmvParams
{
    using value_type = Type;
    static constexpr int32_t grid_2d = Grid2D;
    static constexpr int32_t grid_3d = Grid3D;
    static constexpr int32_t wheel   = Wheel;
    static constexpr int32_t dense   = Dense;
};

// ---------------------------------------------------------
// Test for scan ops taking single input value
// ---------------------------------------------------------

template<class Params>
class HipcubDeviceSpmvTests : public ::testing::Test
{
public:
    using value_type = typename Params::value_type;
    static constexpr int32_t grid_2d = Params::grid_2d;
    static constexpr int32_t grid_3d = Params::grid_3d;
    static constexpr int32_t wheel   = Params::wheel;
    static constexpr int32_t dense   = Params::dense;
};

typedef ::testing::Types<
    DeviceSpmvParams<float, 4, 0, 0, 0>
> HipcubDeviceSpmvTestsParams;

template<typename T, typename OffsetType>
static void
generate_matrix(CooMatrix<T, OffsetType> &coo_matrix,
                int32_t grid2d,
                int32_t grid3d,
                int32_t wheel,
                int32_t dense)
{
    if (grid2d > 0)
    {
        // Generate 2D lattice
        coo_matrix.InitGrid2d(grid2d, false);
    }
    else if (grid3d > 0)
    {
        // Generate 3D lattice
        coo_matrix.InitGrid3d(grid3d, false);
    }
    else if (wheel > 0)
    {
        // Generate wheel graph
        coo_matrix.InitWheel(wheel);
    }
    else if (dense > 0)
    {
        #if 0
        // Generate dense graph
        OffsetType size = 1 << 24; // 16M nnz
        args.GetCmdLineArgument("size", size);

        OffsetType rows = size / dense;
        printf("dense_%d_x_%d, ", rows, dense); fflush(stdout);
        coo_matrix.InitDense(rows, dense);
        #endif
    }
}

template <
    typename T,
    typename OffsetType>
void SpmvGold(
    CsrMatrix<T, OffsetType>&  a,
    T*                         vector_x,
    T*                         vector_y_in,
    T*                         vector_y_out,
    T                          alpha,
    T                          beta)
{
    for (OffsetType row = 0; row < a.num_rows; ++row)
    {
        T partial = beta * vector_y_in[row];
        for (OffsetType offset = a.row_offsets[row];
             offset < a.row_offsets[row + 1];
             ++offset)
        {
            partial += alpha * a.values[offset] * vector_x[a.column_indices[offset]];
        }
        vector_y_out[row] = partial;
    }
}

TYPED_TEST_SUITE(HipcubDeviceSpmvTests, HipcubDeviceSpmvTestsParams);

TYPED_TEST(HipcubDeviceSpmvTests, Spmv)
{
    int device_id = test_common_utils::obtain_device_from_ctest();
    SCOPED_TRACE(testing::Message() << "with device_id= " << device_id);
    HIP_CHECK(hipSetDevice(device_id));

    using T = typename TestFixture::value_type;
    using OffsetType = int32_t;
    constexpr int32_t grid_2d = TestFixture::grid_2d;
    constexpr int32_t grid_3d = TestFixture::grid_3d;
    constexpr int32_t wheel   = TestFixture::wheel;
    constexpr int32_t dense   = TestFixture::dense;

    hipStream_t stream = 0; // default

    CooMatrix<T, OffsetType> coo_matrix;
    generate_matrix(coo_matrix, grid_2d, grid_3d, wheel, dense);

    // Convert to CSR
    CsrMatrix<T, OffsetType> csr_matrix;
    csr_matrix.FromCoo(coo_matrix);

    // Allocate input and output vectors
    T* vector_x        = new T[csr_matrix.num_cols];
    T* vector_y_in     = new T[csr_matrix.num_rows];
    T* vector_y_out    = new T[csr_matrix.num_rows];

    for (int col = 0; col < csr_matrix.num_cols; ++col)
        vector_x[col] = 1.0;

    for (int row = 0; row < csr_matrix.num_rows; ++row)
        vector_y_in[row] = 1.0;

    // Compute reference answer
    SpmvGold(csr_matrix, vector_x, vector_y_in, vector_y_out, alpha, beta);

    // Allocate and initialize GPU problem
    hipcub::DeviceSpmv::SpmvParams<T, OffsetType> params;

    HIP_CHECK(g_allocator.DeviceAllocate((void **) &params.d_values,          sizeof(T) * csr_matrix.num_nonzeros));
    HIP_CHECK(g_allocator.DeviceAllocate((void **) &params.d_row_end_offsets, sizeof(OffsetType) * (csr_matrix.num_rows + 1)));
    HIP_CHECK(g_allocator.DeviceAllocate((void **) &params.d_column_indices,  sizeof(OffsetType) * csr_matrix.num_nonzeros));
    HIP_CHECK(g_allocator.DeviceAllocate((void **) &params.d_vector_x,        sizeof(T) * csr_matrix.num_cols));
    HIP_CHECK(g_allocator.DeviceAllocate((void **) &params.d_vector_y,        sizeof(T) * csr_matrix.num_rows));

    params.num_rows     = csr_matrix.num_rows;
    params.num_cols     = csr_matrix.num_cols;
    params.num_nonzeros = csr_matrix.num_nonzeros;
    params.alpha        = alpha;
    params.beta         = beta;

    HIP_CHECK(hipMemcpy(params.d_values,            csr_matrix.values,         sizeof(T) * csr_matrix.num_nonzeros, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(params.d_row_end_offsets,   csr_matrix.row_offsets,    sizeof(OffsetType) * (csr_matrix.num_rows + 1), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(params.d_column_indices,    csr_matrix.column_indices, sizeof(OffsetType) * csr_matrix.num_nonzeros, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(params.d_vector_x,          vector_x,                  sizeof(T) * csr_matrix.num_cols, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(params.d_vector_y,          vector_y_in,               sizeof(T) * csr_matrix.num_rows, hipMemcpyHostToDevice));

    // Allocate temporary storage
    size_t temp_storage_bytes = 0;
    void *d_temp_storage = NULL;

    // Get amount of temporary storage needed
    HIP_CHECK(hipcub::DeviceSpmv::CsrMV(
                d_temp_storage,
                temp_storage_bytes,
                params.d_values,
                params.d_row_end_offsets,
                params.d_column_indices,
                params.d_vector_x,
                params.d_vector_y,
                params.num_rows,
                params.num_cols,
                params.num_nonzeros,
                stream,
                false));

    // Allocate
    //HIP_CHECK(hipMalloc(&d_temp_storage, temp_storage_bytes);
    HIP_CHECK(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipcub::DeviceSpmv::CsrMV(
                d_temp_storage,
                temp_storage_bytes,
                params.d_values,
                params.d_row_end_offsets,
                params.d_column_indices,
                params.d_vector_x,
                params.d_vector_y,
                params.num_rows,
                params.num_cols,
                params.num_nonzeros,
                stream,
                false));

    HIP_CHECK(hipMemcpy(vector_y_in, params.d_vector_y, sizeof(T) * params.num_rows, hipMemcpyDeviceToHost));

    HIP_CHECK(hipPeekAtLastError());
    HIP_CHECK(hipDeviceSynchronize());

    for(int32_t i = 0; i < csr_matrix.num_rows; i++)
    {
        auto diff = std::max<T>(std::abs(0.01f * vector_y_out[i]), 0.01f);
        ASSERT_NEAR(vector_y_in[i], vector_y_out[i], diff) << "where index = " << i;
    }

}

