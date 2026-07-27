#ifndef RPP_TEST_GENERIC_TENSOR_SETUP_H
#define RPP_TEST_GENERIC_TENSOR_SETUP_H

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include "framework/backend_memory.hpp"
#include "framework/backend_param.hpp"
#include "framework/config_param.hpp"
#include "framework/tensor_setup.hpp"

namespace rpptest {

// Setup for the ND "generic tensor" ops (Misc domain): the binary tensor-vs-tensor
// operations take RpptGenericDesc (numDims/dims[]/strides[]) plus an RpptBroadcastMode,
// rather than the image-domain RpptDesc + XYWH RpptROI. None of the image helpers
// (make_descriptor / make_roi / for_each_roi_io) apply, so the ND axes and traversal
// live here.

// Which operand, if any, is broadcast against the other. The broadcast operand has its
// trailing axis collapsed to extent 1, which is the shape the ops document as
// broadcastable ("for each axis, the corresponding dimensions are either equal or one of
// them is 1"). None => RPP_BROADCAST_DISABLE (both operands the same shape).
enum class Broadcast { None, Src1, Src2 };

inline std::string broadcast_name(Broadcast b) {
    switch (b) {
        case Broadcast::None: return "NoBroadcast";
        case Broadcast::Src1: return "BroadcastSrc1";
        case Broadcast::Src2: return "BroadcastSrc2";
    }
    return "UNK";
}

inline RpptBroadcastMode to_rpp_broadcast(Broadcast b) {
    return b == Broadcast::None ? RPP_BROADCAST_DISABLE : RPP_BROADCAST_ENABLE;
}

// Full tensor extents including the leading batch axis: dims[0] is the batch size and
// dims[1..nDim] the per-sample extents, matching RpptGenericDesc::dims. rank() is the
// per-sample rank (nDim), i.e. what the API's roiTensor is sized against.
using NdDims = std::vector<Rpp32u>;

inline Rpp32u nd_rank(const NdDims& dims) { return static_cast<Rpp32u>(dims.size()) - 1; }

// An input->output dtype pair. Most ND ops keep the dtype (in == out); normalize documents
// genuine conversions (u8->f32, i8->f32), which is why this is an explicit axis.
struct DTypeConv {
    DType in, out;
};

// A single point in the ND test grid: the universal axes only. Op-specific axes (bitwise's
// broadcast mode, normalize's axisMask / compute mode) ride alongside via NdWithParams<P>,
// mirroring the image domain's TestConfig + WithParams<P> split.
//
// The label maps onto the same four structural slots as the image grammar: the rank token takes
// the Layout slot and the op-param token takes the Roi slot.
struct NdConfig {
    RppBackend backend;
    DType dtypeIn, dtypeOut;
    Rpp32u nDim;  // per-sample rank: 2, 3 or 4
};

inline std::string rank_name(Rpp32u nDim) { return std::to_string(nDim) + "D"; }

// Per-rank test extents, small enough to keep the grid fast while keeping every axis
// distinct (so an axis mix-up in strides or broadcasting cannot pass by coincidence).
// The batch axis is included at index 0.
inline NdDims nd_extents(Rpp32u nDim) {
    switch (nDim) {
        case 2:  return {2, 24, 32};
        case 3:  return {2, 5, 12, 16};
        case 4:  return {2, 2, 4, 10, 12};
        default: return {2, 24, 32};
    }
}

// The shape of one operand: the base extents, with the trailing axis collapsed to 1 when
// this operand is the broadcast one.
inline NdDims nd_operand_dims(Rpp32u nDim, Broadcast broadcast, int operand) {
    NdDims dims = nd_extents(nDim);
    const bool collapse = (broadcast == Broadcast::Src1 && operand == 1) ||
                          (broadcast == Broadcast::Src2 && operand == 2);
    if (collapse) dims.back() = 1;
    return dims;
}

// Broadcast result shape: per-axis max of the two operands (the operands only ever differ
// on an axis where one of them is 1).
inline NdDims nd_broadcast_dims(const NdDims& a, const NdDims& b) {
    NdDims out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] > b[i] ? a[i] : b[i];
    return out;
}

inline std::string nd_shape_name(const NdDims& dims) {
    std::string s = std::to_string(dims[0]);
    for (std::size_t i = 1; i < dims.size(); ++i) s += "x" + std::to_string(dims[i]);
    return s;
}

// Produces the value-parameter label for an ND op with no op params, e.g.
// "HOST_U8toF32_3D_2x5x12x16" -- the Roi slot is simply absent, mirroring the image domain's
// config_param_name.
inline std::string nd_config_name(const NdConfig& c) {
    return backend_name(c.backend) + "_" + dtype_name(c.dtypeIn) + "to" + dtype_name(c.dtypeOut) +
           "_" + rank_name(c.nDim) + "_" + nd_shape_name(nd_extents(c.nDim));
}

inline std::string nd_config_param_name(const ::testing::TestParamInfo<NdConfig>& info) {
    return nd_config_name(info.param);
}

// ---- op-specific parameters -----------------------------------------------

template <typename P>
struct NdWithParams {
    NdConfig cfg;
    P op;
};

// Attaches each op-param set to every base config (op params as an extra grid axis).
template <typename P>
inline std::vector<NdWithParams<P>> nd_with_params(const std::vector<NdConfig>& base,
                                                   const std::vector<P>& params) {
    std::vector<NdWithParams<P>> out;
    out.reserve(base.size() * params.size());
    for (const auto& c : base)
        for (const auto& p : params) out.push_back({c, p});
    return out;
}

// Produces the value-parameter label, e.g. "HIP_U8toU8_3D_BroadcastSrc2_2x5x12x16".
// The op-param token sits in the Roi slot, between the rank and the shape.
// P must provide std::string name() const.
template <typename P>
inline std::string nd_config_name(const NdWithParams<P>& p) {
    return backend_name(p.cfg.backend) + "_" + dtype_name(p.cfg.dtypeIn) + "to" +
           dtype_name(p.cfg.dtypeOut) + "_" + rank_name(p.cfg.nDim) + "_" + p.op.name() + "_" +
           nd_shape_name(nd_extents(p.cfg.nDim));
}

template <typename P>
inline std::string nd_op_config_name(const ::testing::TestParamInfo<NdWithParams<P>>& info) {
    return nd_config_name(info.param);
}

// The broadcast mode as an op param, for the binary ND ops.
struct BroadcastParams {
    Broadcast mode;
    std::string name() const { return broadcast_name(mode); }
};

// Cartesian product of the universal ND axes with every available backend (HIP only when
// the suite was built with the HIP backend -- see available_backends()).
inline std::vector<NdConfig> make_nd_configs(const std::vector<DTypeConv>& convs,
                                             const std::vector<Rpp32u>& ranks) {
    std::vector<NdConfig> configs;
    for (RppBackend backend : available_backends())
        for (DTypeConv conv : convs)
            for (Rpp32u nDim : ranks) configs.push_back({backend, conv.in, conv.out, nDim});
    return configs;
}

// Convenience for the ops that keep the dtype (in == out).
inline std::vector<NdConfig> make_nd_configs(const std::vector<DType>& dtypes,
                                             const std::vector<Rpp32u>& ranks) {
    std::vector<DTypeConv> convs;
    convs.reserve(dtypes.size());
    for (DType d : dtypes) convs.push_back({d, d});
    return make_nd_configs(convs, ranks);
}

// ---- descriptor / roiTensor construction ----------------------------------

// Builds a densely packed (row-major, no padding) generic descriptor. numDims counts the
// batch axis, so it is the per-sample rank + 1; strides[i] is the product of all extents
// below axis i, and strides[0] is one sample's element count.
inline RpptGenericDesc make_generic_descriptor(const NdDims& dims, DType dt) {
    RpptGenericDesc d{};
    d.numDims = dims.size();
    d.offsetInBytes = 0;
    d.dataType = to_rpp_dtype(dt);
    d.layout = RpptLayout::NCHW;
    for (std::size_t i = 0; i < dims.size(); ++i) d.dims[i] = dims[i];
    Rpp32u v = 1;
    for (int i = static_cast<int>(d.numDims) - 1; i > 0; --i) {
        d.strides[i] = v;
        v *= d.dims[i];
    }
    d.strides[0] = v;
    return d;
}

// Storage for an RpptGenericDesc, allocated where the backend needs it.
//
// Unlike the image-domain RpptDesc -- which every op reads on the host, so a plain stack
// struct is fine -- the HIP path of the ND ops reads the descriptor's dims/strides *on the
// device* at rank >= 4, so the struct itself must be device-addressable. The legacy HIP
// misc driver does the same (hipHostMalloc of RpptGenericDesc); the requirement is not in
// the API docs and does not apply at rank <= 3.
class GenericDescriptor {
   public:
    GenericDescriptor(RppBackend backend, const NdDims& dims, DType dt) : backend_(backend) {
        if (backend_ == RPP_HIP_BACKEND) {
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
            RPP_TEST_CHECK_HIP(
                hipHostMalloc(reinterpret_cast<void**>(&desc_), sizeof(RpptGenericDesc)));
#endif
        } else {
            desc_ = new RpptGenericDesc();
        }
        *desc_ = make_generic_descriptor(dims, dt);
    }
    ~GenericDescriptor() {
        if (backend_ == RPP_HIP_BACKEND) {
#if defined(RPP_TEST_HAVE_HIP) && RPP_TEST_HAVE_HIP
            (void)hipHostFree(desc_);
#endif
        } else {
            delete desc_;
        }
    }
    GenericDescriptor(const GenericDescriptor&) = delete;
    GenericDescriptor& operator=(const GenericDescriptor&) = delete;

    RpptGenericDescPtr get() const { return desc_; }
    const RpptGenericDesc& operator*() const { return *desc_; }

   private:
    RppBackend backend_;
    RpptGenericDesc* desc_ = nullptr;
};

inline std::size_t generic_element_count(const RpptGenericDesc& d) {
    return static_cast<std::size_t>(d.dims[0]) * d.strides[0];
}

inline std::size_t generic_byte_size(const RpptGenericDesc& d, DType dt) {
    return generic_element_count(d) * dtype_size(dt);
}

// The ops' roiTensor is 2 * nDim values per sample: the per-axis start indices followed by
// the per-axis lengths (the batch axis is not included). The suite exercises the whole
// tensor, so every start is 0 and the lengths are the operand's own extents -- which is
// also what makes the broadcast operand's trailing length 1.
inline std::vector<Rpp32u> make_nd_roi_tensor(const NdDims& dims) {
    const Rpp32u nDim = nd_rank(dims);
    std::vector<Rpp32u> roi(static_cast<std::size_t>(dims[0]) * 2 * nDim, 0);
    for (Rpp32u s = 0; s < dims[0]; ++s) {
        Rpp32u* sample = roi.data() + static_cast<std::size_t>(s) * 2 * nDim;
        for (Rpp32u a = 0; a < nDim; ++a) {
            sample[a] = 0;                 // start
            sample[nDim + a] = dims[a + 1];  // length
        }
    }
    return roi;
}

// ---- traversal ------------------------------------------------------------

// Invokes fn(outIdx, idx1, idx2, coords) for every element of the (broadcast) output.
// An operand axis of extent 1 is held at index 0 while the output axis advances, which is
// the whole of broadcast semantics; every other axis tracks the output coordinate. This is
// the single definition of that mapping, so the reference and the comparator agree.
template <typename Fn>
void for_each_nd_element(const RpptGenericDesc& out, const RpptGenericDesc& s1,
                         const RpptGenericDesc& s2, Fn fn) {
    const std::size_t total = generic_element_count(out);
    const std::size_t rank = out.numDims;
    std::vector<Rpp32u> coord(rank, 0);
    for (std::size_t linear = 0; linear < total; ++linear) {
        std::size_t rem = linear;
        for (std::size_t a = 0; a < rank; ++a) {
            coord[a] = static_cast<Rpp32u>(rem / out.strides[a]);
            rem %= out.strides[a];
        }
        std::size_t idx1 = 0, idx2 = 0;
        for (std::size_t a = 0; a < rank; ++a) {
            idx1 += static_cast<std::size_t>(s1.dims[a] == 1 ? 0 : coord[a]) * s1.strides[a];
            idx2 += static_cast<std::size_t>(s2.dims[a] == 1 ? 0 : coord[a]) * s2.strides[a];
        }
        fn(linear, idx1, idx2, coord);
    }
}

// ---- comparison -----------------------------------------------------------

// Element-wise tolerance comparison over the whole (densely packed) output tensor,
// naming the first offending element by its full ND coordinate.
//
// The bound is absTolerance + relTolerance * |reference|. Ops whose output magnitude varies
// with the input scale (normalize, whose values depend on the supplied mean/stddev) need the
// relative term; bit-exact ops pass 0 for both.
template <typename T>
::testing::AssertionResult compare_nd(const T* actual, const T* reference,
                                      const RpptGenericDesc& d, double absTolerance,
                                      double relTolerance = 0.0) {
    const std::size_t total = generic_element_count(d);
    const std::size_t rank = d.numDims;
    for (std::size_t linear = 0; linear < total; ++linear) {
        const double a = to_double(actual[linear]);
        const double r = to_double(reference[linear]);
        const double diff = std::fabs(a - r);
        const double tolerance = absTolerance + relTolerance * std::fabs(r);
        if (diff > tolerance) {
            std::string coords;
            std::size_t rem = linear;
            for (std::size_t axis = 0; axis < rank; ++axis) {
                coords += (axis ? "," : "") + std::to_string(rem / d.strides[axis]);
                rem %= d.strides[axis];
            }
            return ::testing::AssertionFailure()
                   << "mismatch at [" << coords << "]: actual=" << a << " reference=" << r
                   << " diff=" << diff << " tolerance=" << tolerance;
        }
    }
    return ::testing::AssertionSuccess();
}

}  // namespace rpptest

#endif  // RPP_TEST_GENERIC_TENSOR_SETUP_H
