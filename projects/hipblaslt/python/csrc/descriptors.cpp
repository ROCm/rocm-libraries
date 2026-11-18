// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// Handle, MatmulDesc, MatrixLayout and Preference, with one table-driven attribute API (Review A,
// Decision 3). hipBLASLt checks only that a buffer is at least as big as the attribute, copies
// sizeof(attribute) bytes, and accepts most values unchecked, so the table is what stops a
// wrong-size or out-of-range value from reaching the library.
#include "descriptors.hpp"
#include "init.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>

#include <array>
#include <climits>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>

namespace nb = nanobind;

namespace hipblaslt_py
{

    void require_device()
    {
        int count = 0;
        if(hipGetDeviceCount(&count) != hipSuccess || count <= 0)
            throw HipblasLtError("no HIP device is visible, and hipblasLtCreate would end the "
                                 "process without one (ROCM-33629)");
    }

    Handle::Handle()
    {
        require_device();
        HIPBLASLT_CHECK(hipblasLtCreate(&h_));

        // The loaded-version check needs a handle, so the first one in the process runs it
        // (hipblaslt._loader.check_loaded_version); import never creates one.
        static bool checked = false;
        if(!checked)
        {
            checked       = true;
            int  version  = 0;
            char rev[128] = {};
            if(hipblasLtGetVersion(h_, &version) == HIPBLAS_STATUS_SUCCESS
               && hipblasLtGetGitRevision(h_, rev) == HIPBLAS_STATUS_SUCCESS)
                nb::module_::import_("hipblaslt._loader")
                    .attr("check_loaded_version")(nb::make_tuple(version, std::string(rev)));
        }
    }

    void Handle::close() noexcept
    {
        if(h_)
        {
            hipblasLtDestroy(h_);
            h_ = nullptr;
        }
    }

    hipblasLtHandle_t Handle::raw() const
    {
        if(!h_)
            throw HipblasLtError("this Handle is closed");
        return h_;
    }

    MatmulDesc::MatmulDesc(hipblasComputeType_t compute_type, hipDataType scale_type)
    {
        HIPBLASLT_CHECK(hipblasLtMatmulDescCreate(&d_, compute_type, scale_type));
    }

    MatmulDesc::~MatmulDesc()
    {
        if(d_)
            hipblasLtMatmulDescDestroy(d_);
    }

    MatrixLayout::MatrixLayout(hipDataType dtype, uint64_t rows, uint64_t cols, int64_t ld)
    {
        HIPBLASLT_CHECK(hipblasLtMatrixLayoutCreate(&l_, dtype, rows, cols, ld));
    }

    MatrixLayout::~MatrixLayout()
    {
        if(l_)
            hipblasLtMatrixLayoutDestroy(l_);
    }

    Preference::Preference()
    {
        HIPBLASLT_CHECK(hipblasLtMatmulPreferenceCreate(&p_));
    }

    Preference::~Preference()
    {
        if(p_)
            hipblasLtMatmulPreferenceDestroy(p_);
    }

} // namespace hipblaslt_py

namespace
{
    using namespace hipblaslt_py;

    enum class Kind
    {
        I32,
        U32,
        I64,
        U64,
        F32,
        PTR,
    };

    const char* kind_name(Kind k)
    {
        switch(k)
        {
        case Kind::I32:
            return "int32";
        case Kind::U32:
            return "uint32";
        case Kind::I64:
            return "int64";
        case Kind::U64:
            return "uint64";
        case Kind::F32:
            return "float32";
        case Kind::PTR:
            return "pointer";
        }
        return "?";
    }

    size_t kind_size(Kind k)
    {
        switch(k)
        {
        case Kind::I32:
        case Kind::U32:
        case Kind::F32:
            return 4;
        case Kind::I64:
        case Kind::U64:
            return 8;
        case Kind::PTR:
            return sizeof(void*);
        }
        return 0;
    }

    struct AttrType
    {
        Kind        kind;
        const char* c_enum  = nullptr; // when the value is a member of an enum
        const char* py_enum = nullptr;
    };

    // Types follow hipblaslt.h's comments, except where the implementation
    // (rocblaslt_auxiliary.cpp) stores something else; the implementation wins.
    const std::map<hipblasLtMatmulDescAttributes_t, AttrType>& matmul_desc_types()
    {
        static const std::map<hipblasLtMatmulDescAttributes_t, AttrType> t = {
            {HIPBLASLT_MATMUL_DESC_TRANSA, {Kind::I32, "hipblasOperation_t", "Operation"}},
            {HIPBLASLT_MATMUL_DESC_TRANSB, {Kind::I32, "hipblasOperation_t", "Operation"}},
            // The header says uint32_t; it holds a hipblasLtEpilogue_t, stored as int32_t.
            {HIPBLASLT_MATMUL_DESC_EPILOGUE, {Kind::I32, "hipblasLtEpilogue_t", "Epilogue"}},
            {HIPBLASLT_MATMUL_DESC_BIAS_POINTER, {Kind::PTR}},
            {HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, {Kind::I32, "hipDataType", "DataType"}},
            {HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, {Kind::PTR}},
            {HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, {Kind::PTR}},
            {HIPBLASLT_MATMUL_DESC_C_SCALE_POINTER, {Kind::PTR}},
            {HIPBLASLT_MATMUL_DESC_D_SCALE_POINTER, {Kind::PTR}},
            {HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_SCALE_POINTER, {Kind::PTR}},
            {HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER, {Kind::PTR}},
            {HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD, {Kind::I64}},
            {HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE, {Kind::I64}},
            {HIPBLASLT_MATMUL_DESC_POINTER_MODE,
             {Kind::I32, "hipblasLtPointerMode_t", "PointerMode"}},
            {HIPBLASLT_MATMUL_DESC_AMAX_D_POINTER, {Kind::PTR}},
            {HIPBLASLT_MATMUL_DESC_EPILOGUE_AUX_DATA_TYPE, {Kind::I32, "hipDataType", "DataType"}},
            // The implementation stores int32_t (and rejects negative values).
            {HIPBLASLT_MATMUL_DESC_BIAS_BATCH_STRIDE, {Kind::I32}},
            {HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,
             {Kind::I32, "hipblasLtMatmulMatrixScale_t", "ScaleMode"}},
            {HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,
             {Kind::I32, "hipblasLtMatmulMatrixScale_t", "ScaleMode"}},
            {HIPBLASLT_MATMUL_DESC_SM_COUNT_TARGET, {Kind::I32}},
            {HIPBLASLT_MATMUL_DESC_ALPHA_BATCH_STRIDE, {Kind::I64}},
            {HIPBLASLT_MATMUL_DESC_BETA_BATCH_STRIDE, {Kind::I64}},
            {HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_A_EXT,
             {Kind::I32, "hipDataType", "DataType"}},
            {HIPBLASLT_MATMUL_DESC_COMPUTE_INPUT_TYPE_B_EXT,
             {Kind::I32, "hipDataType", "DataType"}},
            {HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG0_EXT, {Kind::F32}},
            {HIPBLASLT_MATMUL_DESC_EPILOGUE_ACT_ARG1_EXT, {Kind::F32}},
            {HIPBLASLT_MATMUL_DESC_STREAMK_TILE_SCHEDULING_EXT,
             {Kind::I32, "hipblasLtStreamKTileSchedulingMode_t", "StreamKTileSchedulingMode"}},
            {HIPBLASLT_MATMUL_DESC_UNIFORM_SUMMATION_ORDER_EXT, {Kind::I32}},
#if HIPBLASLT_HAS_GEMM_A2A_FUSION
            {HIPBLASLT_MATMUL_DESC_FUSED_EPILOGUE, {Kind::PTR}},
#endif
            {HIPBLASLT_MATMUL_DESC_A_INT4_ENCODING_EXT,
             {Kind::I32, "hipblasLtInt4Encoding_t", "Int4Encoding"}},
        };
        return t;
    }

    const std::map<hipblasLtMatrixLayoutAttribute_t, AttrType>& matrix_layout_types()
    {
        static const std::map<hipblasLtMatrixLayoutAttribute_t, AttrType> t = {
            {HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, {Kind::I32}},
            {HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, {Kind::I64}},
            {HIPBLASLT_MATRIX_LAYOUT_TYPE, {Kind::U32, "hipDataType", "DataType"}},
            {HIPBLASLT_MATRIX_LAYOUT_ORDER, {Kind::I32, "hipblasLtOrder_t", "Order"}},
            {HIPBLASLT_MATRIX_LAYOUT_ROWS, {Kind::U64}},
            {HIPBLASLT_MATRIX_LAYOUT_COLS, {Kind::U64}},
            {HIPBLASLT_MATRIX_LAYOUT_LD, {Kind::I64}},
            // Not in the header; the implementation stores int32_t.
            {HIPBLASLT_MATRIX_LAYOUT_BATCH_MODE, {Kind::I32, "hipblasLtBatchMode_t", "BatchMode"}},
            // Not in the header; the implementation stores int64_t.
            {HIPBLASLT_MATRIX_LAYOUT_OFFSET, {Kind::I64}},
            {HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_ARRAY, {Kind::PTR}},
            {HIPBLASLT_GROUPED_MATRIX_LAYOUT_COLS_ARRAY, {Kind::PTR}},
            {HIPBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY, {Kind::PTR}},
            {HIPBLASLT_GROUPED_MATRIX_LAYOUT_ROWS_COLS_ARRAY_INTEGER_WIDTH,
             {Kind::I32, "hipblasLtIntegerWidth_t", "IntegerWidth"}},
            {HIPBLASLT_GROUPED_MATRIX_LAYOUT_LD_ARRAY_INTEGER_WIDTH,
             {Kind::I32, "hipblasLtIntegerWidth_t", "IntegerWidth"}},
        };
        return t;
    }

    const std::map<hipblasLtMatmulPreferenceAttributes_t, AttrType>& preference_types()
    {
        static const std::map<hipblasLtMatmulPreferenceAttributes_t, AttrType> t = {
            {HIPBLASLT_MATMUL_PREF_SEARCH_MODE, {Kind::U32}},
            // The setter reads 8 bytes whatever size it's given.
            {HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, {Kind::U64}},
            {HIPBLASLT_MATMUL_PREF_SM_COUNT_TARGET, {Kind::I32}},
            {HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_ROWS, {Kind::I64}},
            {HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_AVERAGE_COLS, {Kind::I64}},
            {HIPBLASLT_MATMUL_PREF_GROUPED_AVERAGE_REDUCTION_DIM, {Kind::I64}},
            {HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_ROWS_EXT, {Kind::I64}},
            {HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_COLS_EXT, {Kind::I64}},
            {HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_TOTAL_ROWS_EXT, {Kind::I64}},
            {HIPBLASLT_MATMUL_PREF_GROUPED_DESC_D_MAX_TOTAL_COLS_EXT, {Kind::I64}},
        };
        return t;
    }

    std::string attr_name(nb::handle attr)
    {
        return nb::cast<std::string>(nb::str(attr));
    }

    // An integer Python value, range-checked for the attribute's C type.
    template <typename T>
    T to_integer(nb::handle value, const std::string& name)
    {
        if(!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr()))
            throw nb::type_error((name + " takes an integer, not "
                                  + nb::cast<std::string>(nb::str(value.type().attr("__name__"))))
                                     .c_str());
        bool overflow = false;
        T    out{};
        if constexpr(std::is_signed_v<T>)
        {
            int       of = 0;
            long long v  = PyLong_AsLongLongAndOverflow(value.ptr(), &of);
            overflow
                = of != 0 || v < std::numeric_limits<T>::min() || v > std::numeric_limits<T>::max();
            out = static_cast<T>(v);
        }
        else
        {
            if(PyObject_RichCompareBool(value.ptr(), nb::int_(0).ptr(), Py_LT) == 1)
                overflow = true;
            else
            {
                unsigned long long v = PyLong_AsUnsignedLongLong(value.ptr());
                if(PyErr_Occurred())
                {
                    PyErr_Clear();
                    overflow = true;
                }
                overflow = overflow || v > std::numeric_limits<T>::max();
                out      = static_cast<T>(v);
            }
        }
        if(overflow)
            throw nb::value_error((name + " doesn't fit in " + std::to_string(sizeof(T) * 8)
                                   + (std::is_signed_v<T> ? "-bit signed" : "-bit unsigned")
                                   + " storage: " + nb::cast<std::string>(nb::repr(value)))
                                      .c_str());
        return out;
    }

    std::string a_or_an(const std::string& word)
    {
        return (std::string("AEIOU").find(word[0]) != std::string::npos ? "an " : "a ") + word;
    }

    void check_enum_value(nb::handle value, const AttrType& t, long long v, const std::string& name)
    {
        nb::handle type = value.type();
        if(nb::hasattr(type, "_value2member_map_"))
        {
            std::string got = nb::cast<std::string>(type.attr("__name__"));
            if(got != t.py_enum)
                throw nb::type_error(
                    (name + " takes " + a_or_an(t.py_enum) + ", not " + a_or_an(got)).c_str());
        }
        if(!enum_has_value(t.c_enum, v))
            throw nb::value_error((name + " takes " + a_or_an(t.py_enum) + ", and "
                                   + std::to_string(v) + " isn't one")
                                      .c_str());
    }

    // The bytes for `value`, of exactly the attribute's C size.
    std::pair<std::array<unsigned char, 8>, size_t>
        encode(const AttrType& t, nb::handle value, const std::string& name)
    {
        std::array<unsigned char, 8> buf{};
        auto                         put = [&](auto v) {
            std::memcpy(buf.data(), &v, sizeof(v));
            return std::make_pair(buf, sizeof(v));
        };
        switch(t.kind)
        {
        case Kind::I32:
        {
            int32_t v = to_integer<int32_t>(value, name);
            if(t.c_enum)
                check_enum_value(value, t, v, name);
            return put(v);
        }
        case Kind::U32:
        {
            uint32_t v = to_integer<uint32_t>(value, name);
            if(t.c_enum)
                check_enum_value(value, t, v, name);
            return put(v);
        }
        case Kind::I64:
            return put(to_integer<int64_t>(value, name));
        case Kind::U64:
            return put(to_integer<uint64_t>(value, name));
        case Kind::F32:
        {
            if(!PyFloat_Check(value.ptr())
               && (!PyLong_Check(value.ptr()) || PyBool_Check(value.ptr())))
                throw nb::type_error((name + " takes a number").c_str());
            return put(static_cast<float>(nb::cast<double>(value)));
        }
        case Kind::PTR:
        {
            uintptr_t p = value.is_none() ? 0 : to_integer<uintptr_t>(value, name);
            return put(reinterpret_cast<void*>(p));
        }
        }
        throw nb::value_error("unknown attribute kind");
    }

    nb::object decode(const AttrType& t, const unsigned char* buf)
    {
        auto get = [&](auto v) {
            std::memcpy(&v, buf, sizeof(v));
            return v;
        };
        nb::object out;
        switch(t.kind)
        {
        case Kind::I32:
            out = nb::int_(get(int32_t{}));
            break;
        case Kind::U32:
            out = nb::int_(get(uint32_t{}));
            break;
        case Kind::I64:
            out = nb::int_(get(int64_t{}));
            break;
        case Kind::U64:
            out = nb::int_(get(uint64_t{}));
            break;
        case Kind::F32:
            out = nb::float_(get(float{}));
            break;
        case Kind::PTR:
            out = nb::int_(reinterpret_cast<uintptr_t>(get(static_cast<void*>(nullptr))));
            break;
        }
        if(t.py_enum)
        {
            nb::object enum_type = nb::module_::import_("hipblaslt._core").attr(t.py_enum);
            try
            {
                return enum_type(out);
            }
            catch(nb::python_error&)
            {
                return out; // a value outside the enum (hipBLASLt accepted it unchecked)
            }
        }
        return out;
    }

    template <typename Attr>
    const AttrType& lookup(const std::map<Attr, AttrType>& table, Attr attr, std::string& name)
    {
        name   = attr_name(nb::cast(attr));
        auto i = table.find(attr);
        if(i == table.end())
            throw nb::value_error(
                (name
                 + " has no entry in the bindings' attribute type table, so it can't be set "
                   "safely")
                    .c_str());
        return i->second;
    }

    template <typename Attr>
    std::map<int, std::pair<std::string, std::optional<std::string>>>
        export_table(const std::map<Attr, AttrType>& table)
    {
        std::map<int, std::pair<std::string, std::optional<std::string>>> out;
        for(const auto& [attr, t] : table)
            out[static_cast<int>(attr)]
                = {kind_name(t.kind),
                   t.py_enum ? std::optional<std::string>(t.py_enum) : std::nullopt};
        return out;
    }

    template <typename Obj, typename Attr, typename Set, typename Get>
    void bind_attributes(nb::class_<Obj>& cls,
                         const std::map<Attr, AttrType>& (*table)(),
                         Set set,
                         Get get)
    {
        cls.def(
            "set_attribute",
            [table, set](Obj& self, Attr attr, nb::handle value) {
                std::string     name;
                const AttrType& t = lookup(table(), attr, name);
                auto [buf, size]  = encode(t, value, name);
                std::string call  = "set_attribute(" + name + ")";
                hipblaslt_py::check_status(set(self.raw(), attr, buf.data(), size), call.c_str());
            },
            nb::arg("attr"),
            nb::arg("value").none(),
            "Set an attribute. The value is checked against the attribute's C type, and its enum "
            "if it has one, before hipBLASLt sees it.");
        cls.def(
            "get_attribute",
            [table, get](Obj& self, Attr attr) {
                std::string                  name;
                const AttrType&              t = lookup(table(), attr, name);
                std::array<unsigned char, 8> buf{};
                size_t                       written = 0;
                std::string                  call    = "get_attribute(" + name + ")";
                hipblaslt_py::check_status(
                    get(self.raw(), attr, buf.data(), kind_size(t.kind), &written), call.c_str());
                return decode(t, buf.data());
            },
            nb::arg("attr"),
            "Read an attribute back, as its C type (or enum member) says.");
    }
} // namespace

void init_descriptors(nb::module_& m)
{
    nb::class_<Handle>(m, "Handle", "A hipBLASLt handle. Needs a visible HIP device.")
        .def(nb::init<>())
        .def("close", &Handle::close, "Release the handle. Safe to call more than once.")
        .def_prop_ro("closed", &Handle::closed)
        .def_prop_ro(
            "ptr",
            [](const Handle& h) { return reinterpret_cast<uintptr_t>(h.raw()); },
            "The raw hipblasLtHandle_t, as an integer. Raises once the handle is closed.")
        .def(
            "__enter__", [](Handle& self) -> Handle& { return self; }, nb::rv_policy::reference)
        .def("__exit__", [](Handle& self, nb::args) { self.close(); });

    auto desc
        = nb::class_<MatmulDesc>(m, "MatmulDesc", "A matmul descriptor (hipblasLtMatmulDesc_t).")
              .def(nb::init<hipblasComputeType_t, hipDataType>(),
                   nb::arg("compute_type"),
                   nb::arg("scale_type"))
              .def_prop_ro(
                  "ptr", [](const MatmulDesc& d) { return reinterpret_cast<uintptr_t>(d.raw()); });
    bind_attributes(
        desc, &matmul_desc_types, hipblasLtMatmulDescSetAttribute, hipblasLtMatmulDescGetAttribute);

    auto layout
        = nb::class_<MatrixLayout>(m, "MatrixLayout", "A matrix layout (hipblasLtMatrixLayout_t).")
              .def(nb::init<hipDataType, uint64_t, uint64_t, int64_t>(),
                   nb::arg("dtype"),
                   nb::arg("rows"),
                   nb::arg("cols"),
                   nb::arg("ld"))
              .def_prop_ro("ptr", [](const MatrixLayout& l) {
                  return reinterpret_cast<uintptr_t>(l.raw());
              });
    bind_attributes(layout,
                    &matrix_layout_types,
                    hipblasLtMatrixLayoutSetAttribute,
                    hipblasLtMatrixLayoutGetAttribute);

    auto pref = nb::class_<Preference>(
                    m, "Preference", "A matmul preference (hipblasLtMatmulPreference_t).")
                    .def(nb::init<>())
                    .def_prop_ro("ptr", [](const Preference& p) {
                        return reinterpret_cast<uintptr_t>(p.raw());
                    });
    bind_attributes(pref,
                    &preference_types,
                    hipblasLtMatmulPreferenceSetAttribute,
                    hipblasLtMatmulPreferenceGetAttribute);

    m.def(
        "_attribute_types",
        []() {
            std::map<std::string, std::map<int, std::pair<std::string, std::optional<std::string>>>>
                out;
            out["MatmulDescAttr"]   = export_table(matmul_desc_types());
            out["MatrixLayoutAttr"] = export_table(matrix_layout_types());
            out["PreferenceAttr"]   = export_table(preference_types());
            return out;
        },
        "{attribute enum: {value: (C storage type, enum name or None)}} for the attribute table.");

    m.def(
        "_unchecked_create_status",
        []() {
            hipblasLtHandle_t h = nullptr;
            hipblasStatus_t   s = hipblasLtCreate(&h);
            if(s == HIPBLAS_STATUS_SUCCESS)
                hipblasLtDestroy(h);
            return static_cast<int>(s);
        },
        "Test hook: call hipblasLtCreate with no device check, for the ROCM-33629 regression "
        "test. Without a device it currently ends the process.");
}
