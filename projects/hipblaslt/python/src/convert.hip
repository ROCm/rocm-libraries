// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
// hipblaslt_float8.h uses HIPBLASLT_EXPORT which is defined in hipblaslt-export.h;
// including hipblaslt.h (via status.hpp) ensures all prereqs are in place.
#include "status.hpp"
#include <hipblaslt/hipblaslt_float8.h>
#include <cstdint>
#include <string>
#include <vector>

namespace nb = nanobind;

// Encode one f32 to an 8-bit fp8 byte using the library's own converters, so the
// byte pattern matches what hipBLASLt produces internally.
// The hipblaslt_f8/bf8 structs inherit the float constructor from their base
// classes via `using __hip_fp8_e4m3::__hip_fp8_e4m3;` etc., so hipblaslt_f8(v)
// where v is float is valid and uses the ground-truth conversion path.
// For the FNUZ variants, the float constructor is also available via the same
// `using` directive on __hip_fp8_e4m3_fnuz/__hip_fp8_e5m2_fnuz base classes,
// so direct float construction is used (no _Float16 cast required).
static uint8_t encode_one(float v, const std::string& fmt)
{
    uint8_t b;
    if(fmt == "e4m3")
    {
        hipblaslt_f8 x(v);
        __builtin_memcpy(&b, &x, 1);
        return b;
    }
    if(fmt == "e5m2")
    {
        hipblaslt_bf8 x(v);
        __builtin_memcpy(&b, &x, 1);
        return b;
    }
    if(fmt == "e4m3_fnuz")
    {
        hipblaslt_f8_fnuz x(v);
        __builtin_memcpy(&b, &x, 1);
        return b;
    }
    if(fmt == "e5m2_fnuz")
    {
        hipblaslt_bf8_fnuz x(v);
        __builtin_memcpy(&b, &x, 1);
        return b;
    }
    throw hipblaslt_py::HipblasLtError("pack_fp8: unknown fmt '" + fmt
                                       + "'; expected e4m3, e5m2, e4m3_fnuz, e5m2_fnuz");
}

static float decode_one(uint8_t b, const std::string& fmt)
{
    if(fmt == "e4m3")
    {
        hipblaslt_f8 x;
        __builtin_memcpy(&x, &b, 1);
        return float(x);
    }
    if(fmt == "e5m2")
    {
        hipblaslt_bf8 x;
        __builtin_memcpy(&x, &b, 1);
        return float(x);
    }
    if(fmt == "e4m3_fnuz")
    {
        hipblaslt_f8_fnuz x;
        __builtin_memcpy(&x, &b, 1);
        return float(x);
    }
    if(fmt == "e5m2_fnuz")
    {
        hipblaslt_bf8_fnuz x;
        __builtin_memcpy(&x, &b, 1);
        return float(x);
    }
    throw hipblaslt_py::HipblasLtError("unpack_fp8: unknown fmt '" + fmt
                                       + "'; expected e4m3, e5m2, e4m3_fnuz, e5m2_fnuz");
}

using F32In = nb::ndarray<nb::numpy, float,   nb::c_contig>;
using U8In  = nb::ndarray<nb::numpy, uint8_t, nb::c_contig>;

void init_convert(nb::module_& m)
{
    m.def("pack_fp8",
          [](F32In arr, const std::string& fmt) {
              size_t n    = arr.size();
              auto*  out  = new uint8_t[n];
              const float* src = arr.data();
              for(size_t i = 0; i < n; ++i)
                  out[i] = encode_one(src[i], fmt);
              nb::capsule owner(out, [](void* p) noexcept {
                  delete[] static_cast<uint8_t*>(p);
              });
              std::vector<size_t> shape(arr.shape_ptr(), arr.shape_ptr() + arr.ndim());
              return nb::ndarray<nb::numpy, uint8_t>(out, shape.size(), shape.data(), owner);
          },
          nb::arg("arr"), nb::arg("fmt"),
          "Pack a float32 numpy array to fp8 bytes using hipBLASLt ground-truth converters.\n"
          "fmt: one of 'e4m3' (OCP E4M3), 'e5m2' (OCP E5M2), 'e4m3_fnuz', 'e5m2_fnuz'.");

    m.def("unpack_fp8",
          [](U8In arr, const std::string& fmt) {
              size_t n    = arr.size();
              auto*  out  = new float[n];
              const uint8_t* src = arr.data();
              for(size_t i = 0; i < n; ++i)
                  out[i] = decode_one(src[i], fmt);
              nb::capsule owner(out, [](void* p) noexcept {
                  delete[] static_cast<float*>(p);
              });
              std::vector<size_t> shape(arr.shape_ptr(), arr.shape_ptr() + arr.ndim());
              return nb::ndarray<nb::numpy, float>(out, shape.size(), shape.data(), owner);
          },
          nb::arg("arr"), nb::arg("fmt"),
          "Unpack fp8 bytes to a float32 numpy array using hipBLASLt ground-truth converters.\n"
          "fmt: one of 'e4m3' (OCP E4M3), 'e5m2' (OCP E5M2), 'e4m3_fnuz', 'e5m2_fnuz'.");
}
