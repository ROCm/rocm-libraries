// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The heuristic query and the low-level matmul. Three of the prototype's quiet failures are
// checked here, before hipBLASLt sees the call:
//   - alpha and beta are passed in the descriptor's scale type (the prototype always passed
//     float, so an fp64 GEMM computed garbage);
//   - the workspace must be at least as big as the chosen solution needs, and the size passed to
//     hipBLASLt is always the buffer's own;
//   - each operand must match its layout's data type and be big enough for the layout.
#include "descriptors.hpp"
#include "device_array.hpp"
#include "init.hpp"
#include "status.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace nb = nanobind;
using namespace hipblaslt_py;

namespace
{
    struct Algo
    {
        hipblasLtMatmulAlgo_t algo{};
        // The position in heuristic()'s ranked results, or None for a solution found another
        // way. Not hipBLASLt's catalog index (Review A, Decision 2).
        std::optional<int> heuristic_rank;
    };

    struct HeuristicResult
    {
        Algo   algo;
        size_t workspace_size = 0;
        float  waves_count    = 0.0f;
    };

    std::vector<HeuristicResult> heuristic(Handle&       handle,
                                           MatmulDesc&   desc,
                                           MatrixLayout& a,
                                           MatrixLayout& b,
                                           MatrixLayout& c,
                                           MatrixLayout& d,
                                           Preference&   pref,
                                           int           max_results)
    {
        if(max_results < 1)
            throw nb::value_error("heuristic: max_results must be at least 1");
        std::vector<hipblasLtMatmulHeuristicResult_t> raw(max_results);
        int                                           returned = 0;
        HIPBLASLT_CHECK(hipblasLtMatmulAlgoGetHeuristic(handle.raw(),
                                                        desc.raw(),
                                                        a.raw(),
                                                        b.raw(),
                                                        c.raw(),
                                                        d.raw(),
                                                        pref.raw(),
                                                        max_results,
                                                        raw.data(),
                                                        &returned));
        std::vector<HeuristicResult> out;
        int                          dropped = 0;
        for(int i = 0; i < returned; ++i)
        {
            if(raw[i].state != HIPBLAS_STATUS_SUCCESS)
            {
                ++dropped;
                continue;
            }
            HeuristicResult hr;
            hr.algo.algo           = raw[i].algo;
            hr.algo.heuristic_rank = static_cast<int>(out.size());
            hr.workspace_size      = raw[i].workspaceSize;
            hr.waves_count         = raw[i].wavesCount;
            out.push_back(hr);
        }
        if(dropped)
            nb::module_::import_("logging")
                .attr("getLogger")("hipblaslt")
                .attr("info")("heuristic: dropped %d of %d results whose state wasn't SUCCESS",
                              dropped,
                              returned);
        return out;
    }

    // Storage for one scalar in the descriptor's scale type.
    struct Scalar
    {
        union
        {
            float   f32;
            double  f64;
            int32_t i32;
        };
        const void* ptr() const
        {
            return this;
        }
    };

    Scalar scalar(nb::handle value, hipDataType scale_type, const char* name)
    {
        if(PyBool_Check(value.ptr()) || !(PyFloat_Check(value.ptr()) || PyLong_Check(value.ptr())))
            throw nb::type_error((std::string("matmul: ") + name + " must be a number").c_str());
        Scalar s{};
        switch(scale_type)
        {
        case HIP_R_32F:
            s.f32 = static_cast<float>(nb::cast<double>(value));
            return s;
        case HIP_R_64F:
            s.f64 = nb::cast<double>(value);
            return s;
        case HIP_R_32I:
        {
            double v = nb::cast<double>(value);
            if(v != std::floor(v) || v < std::numeric_limits<int32_t>::min()
               || v > std::numeric_limits<int32_t>::max())
                throw nb::value_error(
                    (std::string("matmul: ") + name + " must be an int32 for an int32 scale type")
                        .c_str());
            s.i32 = static_cast<int32_t>(v);
            return s;
        }
        default:
            throw nb::value_error(("matmul: scale type "
                                   + nb::cast<std::string>(nb::str(nb::cast(scale_type)))
                                   + " isn't supported yet; use R_32F, R_64F or R_32I")
                                      .c_str());
        }
    }

    template <typename T>
    T layout_attr(const MatrixLayout& l, hipblasLtMatrixLayoutAttribute_t attr)
    {
        T      v{};
        size_t written = 0;
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutGetAttribute(l.raw(), attr, &v, sizeof(v), &written));
        return v;
    }

    // The operand's type must be the layout's, and it must hold every element the layout reaches.
    void check_operand(const DeviceArray& arr, const MatrixLayout& l, const char* name)
    {
        auto type
            = static_cast<hipDataType>(layout_attr<uint32_t>(l, HIPBLASLT_MATRIX_LAYOUT_TYPE));
        if(arr.dtype() != type)
            throw nb::value_error((std::string("matmul: ") + name + " is a "
                                   + nb::cast<std::string>(nb::str(nb::cast(arr.dtype())))
                                   + " DeviceArray, but its layout is "
                                   + nb::cast<std::string>(nb::str(nb::cast(type))))
                                      .c_str());
        uint64_t rows   = layout_attr<uint64_t>(l, HIPBLASLT_MATRIX_LAYOUT_ROWS);
        uint64_t cols   = layout_attr<uint64_t>(l, HIPBLASLT_MATRIX_LAYOUT_COLS);
        int64_t  ld     = layout_attr<int64_t>(l, HIPBLASLT_MATRIX_LAYOUT_LD);
        int32_t  batch  = layout_attr<int32_t>(l, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT);
        int64_t  stride = layout_attr<int64_t>(l, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET);
        int32_t  order  = layout_attr<int32_t>(l, HIPBLASLT_MATRIX_LAYOUT_ORDER);
        if(rows == 0 || cols == 0)
            return;
        uint64_t outer = order == HIPBLASLT_ORDER_ROW ? rows : cols;
        uint64_t inner = order == HIPBLASLT_ORDER_ROW ? cols : rows;
        // The last element of the last batch, plus one.
        uint64_t elements = static_cast<uint64_t>(ld) * (outer - 1) + inner;
        if(batch > 1)
            elements += static_cast<uint64_t>(stride) * static_cast<uint64_t>(batch - 1);
        size_t itemsize = host_itemsize(type);
        if(itemsize && elements * itemsize > arr.nbytes())
            throw nb::value_error((std::string("matmul: ") + name + "'s layout reaches "
                                   + std::to_string(elements * itemsize) + " bytes, but its "
                                   + "DeviceArray has " + std::to_string(arr.nbytes()))
                                      .c_str());
    }

    void matmul(Handle&                handle,
                MatmulDesc&            desc,
                nb::handle             alpha,
                DeviceArray&           a,
                MatrixLayout&          la,
                DeviceArray&           b,
                MatrixLayout&          lb,
                nb::handle             beta,
                DeviceArray&           c,
                MatrixLayout&          lc,
                DeviceArray&           d,
                MatrixLayout&          ld,
                const HeuristicResult& result,
                DeviceArray*           workspace,
                uintptr_t              stream_ptr)
    {
        int32_t mode    = 0;
        size_t  written = 0;
        HIPBLASLT_CHECK(hipblasLtMatmulDescGetAttribute(
            desc.raw(), HIPBLASLT_MATMUL_DESC_POINTER_MODE, &mode, sizeof(mode), &written));
        if(mode != HIPBLASLT_POINTER_MODE_HOST)
            throw nb::value_error(
                "matmul: alpha and beta are Python numbers, so POINTER_MODE must be HOST");
        Scalar alpha_s = scalar(alpha, desc.scale_type(), "alpha");
        Scalar beta_s  = scalar(beta, desc.scale_type(), "beta");

        check_operand(a, la, "A");
        check_operand(b, lb, "B");
        check_operand(c, lc, "C");
        check_operand(d, ld, "D");

        size_t have = workspace ? workspace->nbytes() : 0;
        if(have < result.workspace_size)
            throw nb::value_error(("matmul: this solution needs a workspace of "
                                   + std::to_string(result.workspace_size) + " bytes, and "
                                   + (workspace ? "the one given has " + std::to_string(have)
                                                : std::string("none was given")))
                                      .c_str());
        void* ws = workspace && have ? workspace->raw() : nullptr;

        auto stream = reinterpret_cast<hipStream_t>(stream_ptr);
        HIPBLASLT_CHECK(hipblasLtMatmul(handle.raw(),
                                        desc.raw(),
                                        alpha_s.ptr(),
                                        a.raw(),
                                        la.raw(),
                                        b.raw(),
                                        lb.raw(),
                                        beta_s.ptr(),
                                        c.raw(),
                                        lc.raw(),
                                        d.raw(),
                                        ld.raw(),
                                        &result.algo.algo,
                                        ws,
                                        have,
                                        stream));
        // Results are visible to the host when matmul returns.
        HIP_CHECK(hipStreamSynchronize(stream));
    }
} // namespace

void init_matmul(nb::module_& m)
{
    nb::class_<Algo>(m, "Algo", "One hipBLASLt solution, as an opaque hipblasLtMatmulAlgo_t.")
        .def_ro("heuristic_rank",
                &Algo::heuristic_rank,
                "Position in heuristic()'s results (0 is its top pick), or None. Not hipBLASLt's "
                "catalog index.");

    nb::class_<HeuristicResult>(m, "HeuristicResult")
        .def_ro("algo", &HeuristicResult::algo)
        .def_ro("workspace_size",
                &HeuristicResult::workspace_size,
                "Bytes of workspace this solution needs.")
        .def_ro("waves_count", &HeuristicResult::waves_count);

    m.def("heuristic",
          &heuristic,
          nb::arg("handle"),
          nb::arg("desc"),
          nb::arg("a_layout"),
          nb::arg("b_layout"),
          nb::arg("c_layout"),
          nb::arg("d_layout"),
          nb::arg("preference"),
          nb::arg("max_results") = 32,
          "hipBLASLt's recommended solutions for a problem, best first. Results whose state isn't "
          "SUCCESS are dropped, and the count logged.");

    m.def("matmul",
          &matmul,
          nb::arg("handle"),
          nb::arg("desc"),
          nb::arg("alpha"),
          nb::arg("A"),
          nb::arg("a_layout"),
          nb::arg("B"),
          nb::arg("b_layout"),
          nb::arg("beta"),
          nb::arg("C"),
          nb::arg("c_layout"),
          nb::arg("D"),
          nb::arg("d_layout"),
          nb::arg("result"),
          nb::arg("workspace").none() = nb::none(),
          nb::arg("stream")           = 0,
          "D = alpha * op(A) @ op(B) + beta * C, with the solution in `result`. alpha and beta "
          "are Python numbers, passed in the descriptor's scale type. The workspace must be at "
          "least result.workspace_size bytes. Synchronizes the stream before returning.");
}
