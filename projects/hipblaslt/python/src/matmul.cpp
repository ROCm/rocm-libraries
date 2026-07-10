// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <hip/hip_runtime.h>
#include <hipblaslt/hipblaslt.h>
#include <vector>
#include <cstdint>
#include "status.hpp"
#include "descriptors.hpp"
#include "device_array.hpp"
#include "init.hpp"

namespace nb = nanobind;
using namespace hipblaslt_py;

namespace {

struct Algo
{
    hipblasLtMatmulAlgo_t algo;
    int index = -1;   // identifier for logging (0 = fastest from heuristic)
};

struct HeuristicResult
{
    Algo   algo;
    size_t workspace_size = 0;
    float  waves_count = 0.0f;
};

std::vector<HeuristicResult> heuristic(
    Handle& handle, MatmulDesc& desc,
    MatrixLayout& a, MatrixLayout& b, MatrixLayout& c, MatrixLayout& d,
    Preference& pref, int max_results)
{
    std::vector<hipblasLtMatmulHeuristicResult_t> raw(max_results);
    int returned = 0;
    HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(
        handle.raw(), desc.raw(), a.raw(), b.raw(), c.raw(), d.raw(),
        pref.raw(), max_results, raw.data(), &returned));

    std::vector<HeuristicResult> out;
    for(int i = 0; i < returned; ++i)
    {
        if(raw[i].state != HIPBLAS_STATUS_SUCCESS)
            continue;
        HeuristicResult hr;
        hr.algo.algo = raw[i].algo;
        hr.algo.index = i;
        hr.workspace_size = raw[i].workspaceSize;
        hr.waves_count = raw[i].wavesCount;
        out.push_back(hr);
    }
    return out;
}

// alpha/beta are passed as double from Python and cast to float here because
// the common compute type is COMPUTE_32F (fp32 accumulate). If compute type
// is HIPBLAS_COMPUTE_32I the scalars must be int32_t — that path is a known
// limitation and can be added as a follow-up by branching on compute type.
void matmul(Handle& handle, MatmulDesc& desc,
            double alpha, DeviceArray& A, MatrixLayout& la,
            DeviceArray& B, MatrixLayout& lb,
            double beta, DeviceArray& C, MatrixLayout& lc,
            DeviceArray& D, MatrixLayout& ld,
            Algo& algo, DeviceArray& workspace,
            std::uintptr_t stream_ptr)
{
    float alpha_f = static_cast<float>(alpha);
    float beta_f  = static_cast<float>(beta);
    hipStream_t stream = reinterpret_cast<hipStream_t>(stream_ptr);
    HIPBLASLT_CHECK(hipblasLtMatmul(
        handle.raw(), desc.raw(), &alpha_f,
        A.raw(), la.raw(), B.raw(), lb.raw(), &beta_f,
        C.raw(), lc.raw(), D.raw(), ld.raw(),
        &algo.algo, workspace.raw(), workspace.nbytes(), stream));
    // Synchronize so results are visible to the host when this function returns.
    HIP_CHECK(hipStreamSynchronize(stream));
}

} // namespace

void init_matmul(nb::module_& m)
{
    nb::class_<Algo>(m, "Algo")
        .def_ro("index", &Algo::index);

    nb::class_<HeuristicResult>(m, "HeuristicResult")
        .def_ro("algo", &HeuristicResult::algo)
        .def_ro("workspace_size", &HeuristicResult::workspace_size)
        .def_ro("waves_count", &HeuristicResult::waves_count);

    m.def("heuristic", &heuristic,
          nb::arg("handle"), nb::arg("desc"),
          nb::arg("a_layout"), nb::arg("b_layout"),
          nb::arg("c_layout"), nb::arg("d_layout"),
          nb::arg("preference"), nb::arg("max_results") = 32);

    m.def("matmul", &matmul,
          nb::arg("handle"), nb::arg("desc"), nb::arg("alpha"),
          nb::arg("A"), nb::arg("a_layout"), nb::arg("B"), nb::arg("b_layout"),
          nb::arg("beta"), nb::arg("C"), nb::arg("c_layout"),
          nb::arg("D"), nb::arg("d_layout"),
          nb::arg("algo"), nb::arg("workspace"), nb::arg("stream_ptr") = 0);
}
