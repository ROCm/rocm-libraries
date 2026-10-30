// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "tilewright/model.hpp"
#include "tilewright/types.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace {

namespace tw = tilewright;

// Python handle of a loaded model; the engine keeps the model alive while any
// handle or CandidateSet refers to it.
struct PyModel {
  tw::ModelPtr ptr;
};

const char* dtype_name(tw::DataType dt) {
  switch (dt) {
    case tw::DataType::Float: return "Float";
    case tw::DataType::Double: return "Double";
    case tw::DataType::ComplexFloat: return "ComplexFloat";
    case tw::DataType::ComplexDouble: return "ComplexDouble";
    case tw::DataType::Half: return "Half";
    case tw::DataType::Int8x4: return "Int8x4";
    case tw::DataType::Int32: return "Int32";
    case tw::DataType::BFloat16: return "BFloat16";
    case tw::DataType::Int8: return "Int8";
    case tw::DataType::Int4: return "Int4";
    case tw::DataType::Int64: return "Int64";
    case tw::DataType::XFloat32: return "XFloat32";
    case tw::DataType::Float8_fnuz: return "Float8_fnuz";
    case tw::DataType::BFloat8_fnuz: return "BFloat8_fnuz";
    case tw::DataType::Float8BFloat8_fnuz: return "Float8BFloat8_fnuz";
    case tw::DataType::BFloat8Float8_fnuz: return "BFloat8Float8_fnuz";
    case tw::DataType::Float8: return "Float8";
    case tw::DataType::BFloat8: return "BFloat8";
    case tw::DataType::Float8BFloat8: return "Float8BFloat8";
    case tw::DataType::BFloat8Float8: return "BFloat8Float8";
    case tw::DataType::Float6: return "Float6";
    case tw::DataType::BFloat6: return "BFloat6";
    case tw::DataType::Float4: return "Float4";
    case tw::DataType::Count: return "None_";
  }
  return "?";
}

std::string repr(const tw::Dim3& d) {
  return "Dim3(m=" + std::to_string(d.m) + ", n=" + std::to_string(d.n) +
         ", k=" + std::to_string(d.k) + ")";
}

const tw::Model& model_of(const PyModel& m) {
  if (!m.ptr) throw std::invalid_argument("tilewright.Model is empty");
  return *m.ptr;
}

PyModel loaded_or_raise(tw::ModelPtr model, const std::string& error) {
  if (!model) throw std::invalid_argument(error.empty() ? "model load failed" : error);
  return PyModel{std::move(model)};
}

// A dict[str, int] (or None) as Config::attributes, in the dict's order.
std::vector<tw::Attribute> attributes_from(nb::handle value) {
  std::vector<tw::Attribute> out;
  if (value.is_none()) return out;
  if (!nb::isinstance<nb::dict>(value))
    throw nb::type_error("Config.attributes must be a dict of str -> int");
  for (auto [key, item] : nb::borrow<nb::dict>(value)) {
    tw::Attribute a;
    if (!nb::try_cast(key, a.name)) throw nb::type_error("Config.attributes keys must be str");
    if (!nb::try_cast(item, a.value))
      throw nb::type_error(
          ("Config.attributes['" + a.name + "'] must be an int in the int64 range").c_str());
    out.push_back(std::move(a));
  }
  return out;
}

nb::dict attributes_dict(const std::vector<tw::Attribute>& attributes) {
  nb::dict d;
  for (const tw::Attribute& a : attributes)
    d[nb::str(a.name.data(), a.name.size())] = nb::cast(a.value);
  return d;
}

const char* schedule_name(tw::Schedule s) {
  switch (s) {
    case tw::Schedule::Default: return "Default";
    case tw::Schedule::Dynamic: return "Dynamic";
    case tw::Schedule::Auto: return "Auto";
  }
  return "?";
}

const char* reduction_name(tw::PartialReduction r) {
  switch (r) {
    case tw::PartialReduction::None: return "None_";
    case tw::PartialReduction::Tree: return "Tree";
    case tw::PartialReduction::Parallel: return "Parallel";
    case tw::PartialReduction::Atomic: return "Atomic";
  }
  return "?";
}

// A Python callable(config, problem, context) -> LaunchPlan as a LaunchModel,
// with an optional callable(config) -> int as its launch key. Ranking runs
// without the GIL, so each call takes it.
class PyLaunchModel final : public tw::LaunchModel {
 public:
  PyLaunchModel(nb::object fn, nb::object key, std::string id)
      : fn_(std::move(fn)), key_(std::move(key)), id_(std::move(id)) {}
  ~PyLaunchModel() override {
    nb::gil_scoped_acquire gil;
    fn_.reset();
    key_.reset();
  }

  tw::LaunchPlan plan(const tw::Config& config,
                      const tw::Problem& problem,
                      const tw::ExecutionContext& context) const override {
    nb::gil_scoped_acquire gil;
    nb::object out = fn_(nb::cast(config, nb::rv_policy::copy),
                         nb::cast(problem, nb::rv_policy::copy),
                         nb::cast(context, nb::rv_policy::copy));
    tw::LaunchPlan plan;
    if (!nb::try_cast(out, plan))
      throw nb::type_error("a launch model must return a tilewright.LaunchPlan");
    return plan;
  }

  std::uint64_t launch_key(const tw::Config& config) const override {
    if (!key_.is_valid()) return 0;
    nb::gil_scoped_acquire gil;
    nb::object out          = key_(nb::cast(config, nb::rv_policy::copy));
    std::uint64_t key       = 0;
    std::int64_t signed_key = 0;
    if (nb::try_cast(out, key)) return key;
    if (nb::try_cast(out, signed_key)) return static_cast<std::uint64_t>(signed_key);
    throw nb::type_error("a launch key must be an int in [-2**63, 2**64)");
  }

  std::string id() const override { return id_; }

 private:
  nb::object fn_, key_;
  std::string id_;
};

// Plans fixed in advance, by Config::index. Configs with identical plans share
// a launch key (the first index holding that plan), so the table groups
// exactly for the one problem and context it was computed for.
class PlanTable final : public tw::LaunchModel {
 public:
  explicit PlanTable(std::vector<tw::LaunchPlan> plans) : plans_(std::move(plans)) {
    std::map<std::tuple<bool, std::size_t, int>, std::uint64_t> first;
    keys_.reserve(plans_.size());
    for (std::size_t i = 0; i < plans_.size(); ++i) {
      const tw::LaunchPlan& p = plans_[i];
      keys_.push_back(
          first.try_emplace({p.dynamic_queue, p.grid_wgs, static_cast<int>(p.partial_reduction)}, i)
              .first->second);
    }
  }

  tw::LaunchPlan plan(const tw::Config& config,
                      const tw::Problem&,
                      const tw::ExecutionContext&) const override {
    return plans_[checked(config)];
  }

  std::uint64_t launch_key(const tw::Config& config) const override {
    return keys_[checked(config)];
  }

 private:
  std::size_t checked(const tw::Config& config) const {
    if (config.index >= plans_.size())
      throw std::out_of_range("no launch plan for config index " + std::to_string(config.index));
    return config.index;
  }

  std::vector<tw::LaunchPlan> plans_;
  std::vector<std::uint64_t> keys_;
};

std::shared_ptr<const tw::LaunchModel> launch_model_from(nb::handle launch) {
  if (launch.is_none()) return nullptr;
  if (nb::isinstance<nb::list>(launch) || nb::isinstance<nb::tuple>(launch)) {
    std::vector<tw::LaunchPlan> plans;
    if (!nb::try_cast(launch, plans))
      throw nb::type_error("a launch plan list must hold tilewright.LaunchPlan values");
    return std::make_shared<PlanTable>(std::move(plans));
  }
  std::string id;
  if (nb::hasattr(launch, "id")) {
    nb::object value = launch.attr("id");
    if (!value.is_none() && !nb::try_cast(value, id))
      throw nb::type_error("a launch model's id must be a str");
  }
  if (nb::hasattr(launch, "plan")) {
    nb::object key;
    if (nb::hasattr(launch, "launch_key")) key = launch.attr("launch_key");
    return std::make_shared<PyLaunchModel>(
        nb::borrow<nb::object>(launch.attr("plan")), std::move(key), std::move(id));
  }
  if (PyCallable_Check(launch.ptr()))
    return std::make_shared<PyLaunchModel>(
        nb::borrow<nb::object>(launch), nb::object(), std::move(id));
  throw nb::type_error(
      "launch must be None, a callable(config, problem, context) -> LaunchPlan, an object with "
      "such a plan method (and optionally launch_key(config) -> int), or a list of LaunchPlan by "
      "Config.index");
}

}  // namespace

NB_MODULE(_tilewright, m) {
  m.doc() = "GEMM kernel ranking engine (tilewright).";

  nb::enum_<tw::DataType>(m, "DataType", nb::is_arithmetic())
      .value("Float", tw::DataType::Float)
      .value("Double", tw::DataType::Double)
      .value("ComplexFloat", tw::DataType::ComplexFloat)
      .value("ComplexDouble", tw::DataType::ComplexDouble)
      .value("Half", tw::DataType::Half)
      .value("Int8x4", tw::DataType::Int8x4)
      .value("Int32", tw::DataType::Int32)
      .value("BFloat16", tw::DataType::BFloat16)
      .value("Int8", tw::DataType::Int8)
      .value("Int4", tw::DataType::Int4)
      .value("Int64", tw::DataType::Int64)
      .value("XFloat32", tw::DataType::XFloat32)
      .value("Float8_fnuz", tw::DataType::Float8_fnuz)
      .value("BFloat8_fnuz", tw::DataType::BFloat8_fnuz)
      .value("Float8BFloat8_fnuz", tw::DataType::Float8BFloat8_fnuz)
      .value("BFloat8Float8_fnuz", tw::DataType::BFloat8Float8_fnuz)
      .value("Float8", tw::DataType::Float8)
      .value("BFloat8", tw::DataType::BFloat8)
      .value("Float8BFloat8", tw::DataType::Float8BFloat8)
      .value("BFloat8Float8", tw::DataType::BFloat8Float8)
      .value("Float6", tw::DataType::Float6)
      .value("BFloat6", tw::DataType::BFloat6)
      .value("Float4", tw::DataType::Float4)
      .value("None_", tw::DataType::None, "No data type (DataType::None in C++).");

  nb::enum_<tw::Transpose>(m, "Transpose", nb::is_arithmetic())
      .value("T", tw::Transpose::T)
      .value("N", tw::Transpose::N);

  nb::enum_<tw::WeightType>(m, "WeightType", nb::is_arithmetic())
      .value("Fp32", tw::WeightType::Fp32)
      .value("Bf16", tw::WeightType::Bf16)
      .value("Int8", tw::WeightType::Int8)
      .value("Int4", tw::WeightType::Int4);

  nb::enum_<tw::Schedule>(m, "Schedule", nb::is_arithmetic())
      .value("Default", tw::Schedule::Default)
      .value("Dynamic", tw::Schedule::Dynamic)
      .value("Auto", tw::Schedule::Auto);

  nb::enum_<tw::TieBreak>(m, "TieBreak", nb::is_arithmetic())
      .value("PoolOrder", tw::TieBreak::PoolOrder, "Equal scores keep input order.")
      .value("Prior",
             tw::TieBreak::Prior,
             "Equal scores are ordered by ascending prior (non-finite last), then input order.");

  nb::class_<tw::Dim3>(m, "Dim3")
      .def(nb::init<>())
      .def(
          "__init__",
          [](tw::Dim3* self, std::size_t mm, std::size_t nn, std::size_t kk) {
            new (self) tw::Dim3{mm, nn, kk};
          },
          "m"_a,
          "n"_a,
          "k"_a)
      .def_rw("m", &tw::Dim3::m)
      .def_rw("n", &tw::Dim3::n)
      .def_rw("k", &tw::Dim3::k)
      .def("mn", &tw::Dim3::mn)
      .def("mk", &tw::Dim3::mk)
      .def("nk", &tw::Dim3::nk)
      .def("__eq__",
           [](const tw::Dim3& a, const tw::Dim3& b) {
             return a.m == b.m && a.n == b.n && a.k == b.k;
           })
      .def("__repr__", [](const tw::Dim3& d) { return repr(d); });

  nb::class_<tw::Problem>(m, "Problem")
      .def(
          "__init__",
          [](tw::Problem* self,
             tw::Dim3 size,
             std::size_t batch,
             tw::Transpose a_transpose,
             tw::Transpose b_transpose,
             tw::DataType a_dtype,
             tw::DataType b_dtype,
             tw::DataType c_dtype,
             tw::DataType d_dtype,
             tw::DataType mi_dtype) {
            new (self) tw::Problem{size,
                                   batch,
                                   a_transpose,
                                   b_transpose,
                                   a_dtype,
                                   b_dtype,
                                   c_dtype,
                                   d_dtype,
                                   mi_dtype};
          },
          "size"_a        = tw::Dim3{},
          "batch"_a       = std::size_t{1},
          "a_transpose"_a = tw::Transpose::N,
          "b_transpose"_a = tw::Transpose::N,
          "a_dtype"_a     = tw::DataType::None,
          "b_dtype"_a     = tw::DataType::None,
          "c_dtype"_a     = tw::DataType::None,
          "d_dtype"_a     = tw::DataType::None,
          "mi_dtype"_a    = tw::DataType::None)
      .def_rw("size", &tw::Problem::size)
      .def_rw("batch", &tw::Problem::batch)
      .def_rw("a_transpose", &tw::Problem::a_transpose)
      .def_rw("b_transpose", &tw::Problem::b_transpose)
      .def_rw("a_dtype", &tw::Problem::a_dtype)
      .def_rw("b_dtype", &tw::Problem::b_dtype)
      .def_rw("c_dtype", &tw::Problem::c_dtype)
      .def_rw("d_dtype", &tw::Problem::d_dtype)
      .def_rw("mi_dtype", &tw::Problem::mi_dtype)
      .def("__repr__", [](const tw::Problem& p) {
        return "Problem(size=" + repr(p.size) + ", batch=" + std::to_string(p.batch) +
               ", a_transpose=" + (p.a_transpose == tw::Transpose::T ? "T" : "N") +
               ", b_transpose=" + (p.b_transpose == tw::Transpose::T ? "T" : "N") +
               ", a_dtype=" + dtype_name(p.a_dtype) + ", b_dtype=" + dtype_name(p.b_dtype) +
               ", c_dtype=" + dtype_name(p.c_dtype) + ", d_dtype=" + dtype_name(p.d_dtype) +
               ", mi_dtype=" + dtype_name(p.mi_dtype) + ")";
      });

  nb::class_<tw::Config>(m, "Config")
      .def(
          "__init__",
          [](tw::Config* self,
             tw::Dim3 mt,
             tw::Dim3 mi,
             int occupancy,
             int cache_hints_a,
             int cache_hints_b,
             std::size_t grvw_a,
             std::size_t grvw_b,
             std::size_t gwvw_d,
             std::size_t index,
             nb::handle attributes) {
            std::vector<tw::Attribute> attrs = attributes_from(attributes);
            tw::Config* c                    = new (self) tw::Config();
            c->mt                            = mt;
            c->mi                            = mi;
            c->occupancy                     = occupancy;
            c->cache_hints_a                 = cache_hints_a;
            c->cache_hints_b                 = cache_hints_b;
            c->grvw_a                        = grvw_a;
            c->grvw_b                        = grvw_b;
            c->gwvw_d                        = gwvw_d;
            c->index                         = index;
            c->attributes                    = std::move(attrs);
          },
          "mt"_a            = tw::Dim3{},
          "mi"_a            = tw::Dim3{},
          "occupancy"_a     = -1,
          "cache_hints_a"_a = 0,
          "cache_hints_b"_a = 0,
          "grvw_a"_a        = std::size_t{1},
          "grvw_b"_a        = std::size_t{1},
          "gwvw_d"_a        = std::size_t{1},
          "index"_a         = std::size_t{0},
          "attributes"_a    = nb::none(),
          "attributes: dict[str, int] of further kernel properties (names must be non-empty).")
      .def_rw("mt", &tw::Config::mt)
      .def_rw("mi", &tw::Config::mi)
      .def_rw("occupancy", &tw::Config::occupancy)
      .def_rw("cache_hints_a", &tw::Config::cache_hints_a)
      .def_rw("cache_hints_b", &tw::Config::cache_hints_b)
      .def_rw("grvw_a", &tw::Config::grvw_a)
      .def_rw("grvw_b", &tw::Config::grvw_b)
      .def_rw("gwvw_d", &tw::Config::gwvw_d)
      .def_rw("index", &tw::Config::index)
      .def_prop_rw(
          "attributes",
          [](const tw::Config& c) { return attributes_dict(c.attributes); },
          [](tw::Config& c, nb::handle value) { c.attributes = attributes_from(value); },
          nb::for_setter(nb::arg("value").none()),
          "dict[str, int]; a copy, so assign a new dict to change it.")
      .def("__repr__", [](const tw::Config& c) {
        std::string s =
            "Config(mt=" + repr(c.mt) + ", mi=" + repr(c.mi) +
            ", occupancy=" + std::to_string(c.occupancy) +
            ", cache_hints_a=" + std::to_string(c.cache_hints_a) +
            ", cache_hints_b=" + std::to_string(c.cache_hints_b) +
            ", grvw_a=" + std::to_string(c.grvw_a) + ", grvw_b=" + std::to_string(c.grvw_b) +
            ", gwvw_d=" + std::to_string(c.gwvw_d) + ", index=" + std::to_string(c.index);
        if (!c.attributes.empty())
          s += ", attributes=" + std::string(nb::repr(attributes_dict(c.attributes)).c_str());
        return s + ")";
      });

  nb::class_<tw::Hardware>(m, "Hardware")
      .def(
          "__init__",
          [](tw::Hardware* self, std::size_t n_cu, std::size_t lds, std::size_t l2) {
            new (self) tw::Hardware{n_cu, lds, l2};
          },
          "N_CU"_a         = std::size_t{0},
          "lds_capacity"_a = std::size_t{0},
          "L2_capacity"_a  = std::size_t{0})
      .def_rw("N_CU", &tw::Hardware::N_CU)
      .def_rw("lds_capacity", &tw::Hardware::lds_capacity)
      .def_rw("L2_capacity", &tw::Hardware::L2_capacity)
      .def("__repr__", [](const tw::Hardware& h) {
        return "Hardware(N_CU=" + std::to_string(h.N_CU) +
               ", lds_capacity=" + std::to_string(h.lds_capacity) +
               ", L2_capacity=" + std::to_string(h.L2_capacity) + ")";
      });

  nb::class_<tw::ExecutionContext>(m,
                                   "ExecutionContext",
                                   "Where the ranked kernels will run; the defaults describe "
                                   "exclusive use of the device.")
      .def(
          "__init__",
          [](tw::ExecutionContext* self, std::size_t cu_budget, tw::Schedule schedule) {
            new (self) tw::ExecutionContext{cu_budget, schedule};
          },
          "cu_budget"_a = std::size_t{0},
          "schedule"_a  = tw::Schedule::Default,
          "cu_budget: CUs the kernels may use (0, or at least Hardware.N_CU, means every CU).")
      .def_rw("cu_budget", &tw::ExecutionContext::cu_budget)
      .def_rw("schedule", &tw::ExecutionContext::schedule)
      .def("__repr__", [](const tw::ExecutionContext& c) {
        return "ExecutionContext(cu_budget=" + std::to_string(c.cu_budget) +
               ", schedule=" + schedule_name(c.schedule) + ")";
      });

  nb::enum_<tw::PartialReduction>(m, "PartialReduction", nb::is_arithmetic())
      .value("None_", tw::PartialReduction::None, "No split tile (PartialReduction::None in C++).")
      .value("Tree", tw::PartialReduction::Tree)
      .value("Parallel", tw::PartialReduction::Parallel)
      .value("Atomic", tw::PartialReduction::Atomic);

  nb::class_<tw::LaunchPlan>(
      m,
      "LaunchPlan",
      "How a kernel is launched for one problem; the default is a static launch of one "
      "workgroup per output tile.")
      .def(
          "__init__",
          [](tw::LaunchPlan* self,
             bool dynamic_queue,
             std::size_t grid_wgs,
             tw::PartialReduction partial_reduction) {
            new (self) tw::LaunchPlan{dynamic_queue, grid_wgs, partial_reduction};
          },
          "dynamic_queue"_a     = false,
          "grid_wgs"_a          = std::size_t{0},
          "partial_reduction"_a = tw::PartialReduction::None,
          "grid_wgs: workgroups launched (0 means one per output tile).")
      .def_rw("dynamic_queue", &tw::LaunchPlan::dynamic_queue)
      .def_rw("grid_wgs", &tw::LaunchPlan::grid_wgs)
      .def_rw("partial_reduction", &tw::LaunchPlan::partial_reduction)
      .def("__eq__",
           [](const tw::LaunchPlan& a, const tw::LaunchPlan& b) {
             return a.dynamic_queue == b.dynamic_queue && a.grid_wgs == b.grid_wgs &&
                    a.partial_reduction == b.partial_reduction;
           })
      .def("__repr__", [](const tw::LaunchPlan& p) {
        return std::string("LaunchPlan(dynamic_queue=") + (p.dynamic_queue ? "True" : "False") +
               ", grid_wgs=" + std::to_string(p.grid_wgs) +
               ", partial_reduction=PartialReduction." + reduction_name(p.partial_reduction) + ")";
      });

  nb::class_<tw::Result>(m, "Result")
      .def(nb::init<>())
      .def_rw("config_index", &tw::Result::config_index)
      .def_rw("score", &tw::Result::score)
      .def_rw("scored", &tw::Result::scored)
      .def("__eq__",
           [](const tw::Result& a, const tw::Result& b) {
             return a.config_index == b.config_index && a.scored == b.scored && a.score == b.score;
           })
      .def("__repr__", [](const tw::Result& r) {
        return "Result(config_index=" + std::to_string(r.config_index) +
               ", score=" + std::to_string(r.score) + ", scored=" + (r.scored ? "True" : "False") +
               ")";
      });

  nb::class_<tw::Provenance>(m,
                             "Provenance",
                             "Where an MLREC_v3 model came from; a field its file does not "
                             "record is empty.")
      .def_ro("config_id", &tw::Provenance::config_id)
      .def_ro("training_data_hash", &tw::Provenance::training_data_hash)
      .def_ro("source_revision", &tw::Provenance::source_revision)
      .def_ro("date", &tw::Provenance::date)
      .def_ro("launch_model", &tw::Provenance::launch_model)
      .def("__repr__", [](const tw::Provenance& p) {
        return "Provenance(config_id='" + p.config_id + "', training_data_hash='" +
               p.training_data_hash + "', source_revision='" + p.source_revision + "', date='" +
               p.date + "', launch_model='" + p.launch_model + "')";
      });

  nb::class_<tw::AlignmentVeto>(m,
                                "AlignmentVeto",
                                "An MLREC_v3 serving rule: for a layout with exactly one "
                                "K-contiguous operand and a K above k_min whose elements of that "
                                "operand leave a partial align_bits-bit line, kernels whose cache "
                                "hint on it is the non-temporal one take no part in a ranking "
                                "while another kernel can.")
      .def_ro("on", &tw::AlignmentVeto::on)
      .def_ro("k_min", &tw::AlignmentVeto::k_min)
      .def_ro("align_bits", &tw::AlignmentVeto::align_bits)
      .def("__repr__", [](const tw::AlignmentVeto& v) {
        return "AlignmentVeto(on=" + std::string(v.on ? "True" : "False") +
               ", k_min=" + std::to_string(v.k_min) +
               ", align_bits=" + std::to_string(v.align_bits) + ")";
      });

  nb::class_<tw::ModelInfo>(m, "ModelInfo")
      .def_ro("arch", &tw::ModelInfo::arch)
      .def_ro("feature_catalog_hash", &tw::ModelInfo::feature_catalog_hash)
      .def_ro("weight_type", &tw::ModelInfo::weight_type)
      .def_ro("n_cells", &tw::ModelInfo::n_cells)
      .def_ro("n_splits", &tw::ModelInfo::n_splits)
      .def_ro("format_version", &tw::ModelInfo::format_version)
      .def_ro("feature_spec_hash", &tw::ModelInfo::feature_spec_hash)
      .def_ro("query_dim", &tw::ModelInfo::query_dim)
      .def_ro("item_dim", &tw::ModelInfo::item_dim)
      .def_ro("inter_dim", &tw::ModelInfo::inter_dim)
      .def_ro("min_cu_fraction", &tw::ModelInfo::min_cu_fraction)
      .def_ro("max_cu_fraction", &tw::ModelInfo::max_cu_fraction)
      .def_ro("schedules", &tw::ModelInfo::schedules, "A bit per Schedule value.")
      .def_ro("min_effective_cus",
              &tw::ModelInfo::min_effective_cus,
              "The fewest usable CUs the model supports on any device (MLREC_v3), 0 for none.")
      .def_ro("cascade_keys", &tw::ModelInfo::cascade_keys)
      .def_ro("whitelist_tier", &tw::ModelInfo::whitelist_tier)
      .def_ro("launch_keys",
              &tw::ModelInfo::launch_keys,
              "Whether the pool is grouped by the launch model's launch_key.")
      .def_ro("small_batched_rule", &tw::ModelInfo::small_batched_rule)
      .def_ro("dot2_rule", &tw::ModelInfo::dot2_rule)
      .def_ro("non_temporal_rule", &tw::ModelInfo::non_temporal_rule)
      .def_ro("non_temporal_family_rule",
              &tw::ModelInfo::non_temporal_family_rule,
              "Whether a kernel family the non-temporal rule rejects whole keeps one signature, "
              "with every kernel of it, in the rule's region (MLREC_v3).")
      .def_ro("alignment_veto",
              &tw::ModelInfo::alignment_veto,
              "The alignment veto (MLREC_v3); `on` is False for a model without one.")
      .def_ro("clip", &tw::ModelInfo::clip)
      .def_ro("bank",
              &tw::ModelInfo::bank,
              "Whether the model was trained on the integration's bank as well as its table.")
      .def_ro("trained_key_gate",
              &tw::ModelInfo::trained_key_gate,
              "Whether only kernels whose view key had training data are scored while one of "
              "them is feasible.")
      .def_ro("trained_keys",
              &tw::ModelInfo::trained_keys,
              "The number of view keys the trained-key gate lists.")
      .def_ro("drift_keep_ppm",
              &tw::ModelInfo::drift_keep_ppm,
              "The drift fallback's share of a cell's trained whitelisted view keys, in "
              "millionths, that a pool must hold for the cell to be served from its whitelist "
              "alone (0: no drift fallback).")
      .def_ro("kernel_guards",
              &tw::ModelInfo::kernel_guards,
              "Whether the model was trained with the integration's kernel guards, which it "
              "is served with.")
      .def_ro("uncovered_base_labels", &tw::ModelInfo::uncovered_base_labels)
      .def_ro("packed_bytes", &tw::ModelInfo::packed_bytes, "Bytes of the packed weights.")
      .def_ro("expanded_bytes",
              &tw::ModelInfo::expanded_bytes,
              "Bytes of the fp32 weights of every cell (the cache holds those in use).")
      .def_ro("provenance", &tw::ModelInfo::provenance)
      .def("__repr__", [](const tw::ModelInfo& i) {
        return "ModelInfo(arch='" + i.arch + "', feature_catalog_hash='" + i.feature_catalog_hash +
               "', format_version=" + std::to_string(i.format_version) +
               ", weight_type=" + std::to_string(static_cast<int>(i.weight_type)) +
               ", n_cells=" + std::to_string(i.n_cells) +
               ", n_splits=" + std::to_string(i.n_splits) + ")";
      });

  nb::class_<tw::Features>(m, "Features")
      .def_ro("query", &tw::Features::query)
      .def_ro("item", &tw::Features::item)
      .def_ro("interaction", &tw::Features::interaction);

  nb::class_<PyModel>(m, "Model", "A loaded model; create one with load_model*().")
      .def("describe", [](const PyModel& self) { return tw::describe(model_of(self)); })
      .def(
          "route",
          [](const PyModel& self, const tw::Problem& p) { return tw::route(model_of(self), p); },
          "problem"_a)
      .def(
          "cell_label",
          [](const PyModel& self, int cell) { return tw::cell_label(model_of(self), cell); },
          "cell"_a)
      .def("__repr__", [](const PyModel& self) {
        if (!self.ptr) return std::string("Model(<empty>)");
        const tw::ModelInfo i = tw::describe(*self.ptr);
        return "Model(arch='" + i.arch + "', n_cells=" + std::to_string(i.n_cells) + ")";
      });

  nb::class_<tw::CandidateSet>(m, "CandidateSet")
      .def(
          "__init__",
          [](tw::CandidateSet* self,
             const PyModel& model,
             std::vector<tw::Config> configs,
             tw::TieBreak tie_break,
             nb::handle launch) {
            model_of(model);
            tw::PoolOptions options{tie_break, launch_model_from(launch)};
            new (self) tw::CandidateSet(model.ptr, std::move(configs), std::move(options));
          },
          "model"_a,
          "configs"_a,
          "tie_break"_a = tw::TieBreak::PoolOrder,
          "launch"_a    = nb::none(),
          "launch: the integration's launch model (PoolOptions::launch): a callable(config, "
          "problem, context) -> LaunchPlan (launch key 0), an object with such a plan method and "
          "optionally launch_key(config) -> int, or a list of LaunchPlan by Config.index (for one "
          "problem and context; configs with identical plans share a launch key); None gives "
          "every config LaunchPlan() and launch key 0. Configs with equal signature, spec "
          "attributes and launch key must get equal plans. Its errors propagate out of the call "
          "that asked. A callable or object may define a str `id` (LaunchModel::id; absent or "
          "None means empty). Raises ValueError for a config with an empty or duplicate attribute "
          "name, or when the model's provenance names a launch model and `launch` is None or has "
          "another id.")
      .def(
          "rank",
          [](const tw::CandidateSet& self,
             tw::Problem p,
             tw::Hardware h,
             std::size_t min_scored,
             std::optional<tw::ExecutionContext> context,
             std::optional<std::vector<double>> prior) {
            nb::gil_scoped_release release;
            return self.rank(p,
                             h,
                             context.value_or(tw::ExecutionContext{}),
                             min_scored,
                             prior ? &*prior : nullptr);
          },
          "problem"_a,
          "hardware"_a,
          "min_scored"_a = std::size_t{0},
          "context"_a    = nb::none(),
          "prior"_a      = nb::none(),
          "One Result per config: whitelist survivors best-first, then (when min_scored "
          "exceeds their count) the remaining feasible configs best-first, then unscored "
          "configs in input order; in a cell degraded_cells() lists, both are ranked "
          "together, best-first. context None means exclusive use of the device; a context "
          "is ranked at scoring_context(), and every config is unscored for a schedule the "
          "model does not list. With "
          "TieBreak.Prior, prior (one float per config) orders equal scores; ValueError "
          "when it has the wrong length.")
      .def(
          "rank_top",
          [](const tw::CandidateSet& self,
             tw::Problem p,
             tw::Hardware h,
             std::size_t k,
             std::optional<tw::ExecutionContext> context,
             std::optional<std::vector<double>> prior) {
            nb::gil_scoped_release release;
            return self.rank_top(
                p, h, context.value_or(tw::ExecutionContext{}), k, prior ? &*prior : nullptr);
          },
          "problem"_a,
          "hardware"_a,
          "k"_a,
          "context"_a = nb::none(),
          "prior"_a   = nb::none(),
          "The best k configs: the scored results among the first k of rank(problem, "
          "hardware, k, context, prior), fewer when fewer are scored.")
      .def(
          "prepare",
          [](const tw::CandidateSet& self,
             std::optional<std::vector<int>> cells,
             std::optional<tw::Problem> problem,
             std::optional<tw::Hardware> hardware) {
            if (problem.has_value() != hardware.has_value())
              throw std::invalid_argument("prepare takes a problem and a hardware together");
            nb::gil_scoped_release release;
            if (problem) self.prepare(*problem, *hardware);
            if (cells) self.prepare(*cells);
            if (!problem && !cells) self.prepare();
          },
          "cells"_a    = nb::none(),
          "problem"_a  = nb::none(),
          "hardware"_a = nb::none(),
          "Build now what rankings would build on first use: for the given cells (every cell "
          "when neither cells nor a problem is given; the cell the problem routes to), their "
          "fp32 weights, whitelist membership and item embeddings of every config; with a "
          "problem and hardware, also the kernel-only interaction features for its data types.")
      .def_prop_ro("configs", [](const tw::CandidateSet& s) { return s.configs(); })
      .def_prop_ro("untrained_keys",
                   &tw::CandidateSet::untrained_keys,
                   "Groups of configs the model's trained-key gate leaves out while a trained "
                   "one is feasible (0 without a gate).")
      .def_prop_ro(
          "untrained_configs", &tw::CandidateSet::untrained_configs, "The configs of those groups.")
      .def("degraded_cells",
           &tw::CandidateSet::degraded_cells,
           "The cells (indices as route returns them, ascending) the model's drift fallback "
           "serves degraded on this pool; empty without the fallback.")
      .def("__len__", [](const tw::CandidateSet& s) { return s.configs().size(); });

  m.def(
      "feature_catalog_hash",
      []() { return std::string(tw::feature_catalog_hash()); },
      "Hash of the MLREC_v2 feature catalog.");

  m.def("feature_catalog_hashes",
        &tw::feature_catalog_hashes,
        "Hashes of every feature catalog the engine computes (MLREC_v2, then MLREC_v3).");

  m.def(
      "load_model",
      [](const std::string& path) {
        std::string error;
        tw::ModelPtr model;
        {
          nb::gil_scoped_release release;
          model = tw::load_model(path, &error);
        }
        return loaded_or_raise(std::move(model), error);
      },
      "path"_a,
      "Load a model file. Raises ValueError with the reason when it is not a valid model.");

  m.def(
      "load_model_from_memory",
      [](nb::bytes data) {
        const void* ptr        = data.c_str();
        const std::size_t size = data.size();
        std::string error;
        tw::ModelPtr model;
        {
          nb::gil_scoped_release release;
          model = tw::load_model_from_memory(ptr, size, &error);
        }
        return loaded_or_raise(std::move(model), error);
      },
      "data"_a,
      "Load a model from the bytes of a model file. Raises ValueError when they are invalid.");

  m.def(
      "load_model_by_index",
      [](const std::string& logic_stem,
         const std::string& dir,
         std::uint32_t pci_chip_id) -> nb::object {
        std::string error;
        tw::ModelPtr model;
        {
          nb::gil_scoped_release release;
          model = tw::load_model_by_index(logic_stem, dir, pci_chip_id, &error);
        }
        if (model) return nb::cast(PyModel{std::move(model)});
        if (!error.empty()) throw std::invalid_argument(error);
        return nb::none();
      },
      "logic_stem"_a,
      "dir"_a,
      "pci_chip_id"_a = std::uint32_t{0},
      "Load the model <dir>/tilewright_index lists for logic_stem: its row naming pci_chip_id, "
      "else its row without a chip ID (0: unknown, that row only). Returns None when the index "
      "or such a row is absent; raises ValueError when the listed model fails to load or a row "
      "of the stem is malformed.");

  m.def("describe", [](const PyModel& model) { return tw::describe(model_of(model)); }, "model"_a);

  m.def(
      "route",
      [](const PyModel& model, const tw::Problem& p) { return tw::route(model_of(model), p); },
      "model"_a,
      "problem"_a,
      "Index of the cell that scores the problem, or -1.");

  m.def(
      "cell_label",
      [](const PyModel& model, int cell) { return tw::cell_label(model_of(model), cell); },
      "model"_a,
      "cell"_a);

  m.def(
      "compute_features",
      [](const PyModel& model,
         const tw::Problem& p,
         const tw::Config& c,
         const tw::Hardware& h,
         std::optional<tw::ExecutionContext> context,
         std::optional<tw::LaunchPlan> plan) {
        return tw::compute_features(model_of(model),
                                    p,
                                    c,
                                    h,
                                    context.value_or(tw::ExecutionContext{}),
                                    plan.value_or(tw::LaunchPlan{}));
      },
      "model"_a,
      "problem"_a,
      "config"_a,
      "hardware"_a,
      "context"_a = nb::none(),
      "plan"_a    = nb::none(),
      "Raw (unwhitened) query, item and interaction features for the context (None: exclusive "
      "use of the device) and the config's launch plan (None: LaunchPlan()); empty for invalid "
      "hardware.");

  m.def(
      "attribute_names",
      [](const PyModel& model) { return tw::attribute_names(model_of(model)); },
      "model"_a,
      "Names of the Config attributes the model reads (v2 models: none); a config without one "
      "is ranked with the model's default value.");

  m.def(
      "set_cache_limit",
      [](std::size_t bytes) { tw::set_cache_limit(bytes); },
      "bytes"_a,
      "Limit the bytes of fp32 cell weights and candidate-set caches the process keeps; the "
      "least recently used are released past it and rebuilt when needed, with the same "
      "results.");
  m.def("cache_limit", []() { return tw::cache_limit(); }, "The cache limit in bytes.");
  m.def("cache_usage", []() { return tw::cache_usage(); }, "Bytes the cache holds now.");

  m.def(
      "supports",
      [](const PyModel& model, const tw::ExecutionContext& context, const tw::Hardware& h) {
        return tw::supports(model_of(model), context, h);
      },
      "model"_a,
      "context"_a,
      "hardware"_a,
      "True when the model declares the context on this hardware (v2 models: every CU "
      "and Schedule.Default; v3 models: the schedules and CU fractions they declare, and at "
      "least the usable CUs of their CU floor).");

  m.def(
      "scoring_context",
      [](const PyModel& model, const tw::ExecutionContext& context, const tw::Hardware& h) {
        return tw::scoring_context(model_of(model), context, h);
      },
      "model"_a,
      "context"_a,
      "hardware"_a,
      "The context a ranking for `context` is scored at: the context itself when supports() "
      "holds; for a context outside it only by its CU budget, the same schedule with the "
      "nearest usable CUs the model supports on this hardware or, on hardware where it "
      "supports none (fewer CUs than the CU floor), the most its largest CU fraction allows "
      "(cu_budget 0 for every CU); None for a context supports() rejects with a schedule the "
      "model does not list or on hardware with no CUs.");

  m.def(
      "rank_configs",
      [](const PyModel& model,
         tw::Problem p,
         tw::Hardware h,
         const std::vector<tw::Config>& configs,
         std::size_t min_scored,
         std::optional<tw::ExecutionContext> context,
         tw::TieBreak tie_break,
         std::optional<std::vector<double>> prior,
         nb::handle launch) {
        const tw::Model& mdl = model_of(model);
        const tw::PoolOptions options{tie_break, launch_model_from(launch)};
        nb::gil_scoped_release release;
        return tw::rank_configs(mdl,
                                p,
                                h,
                                context.value_or(tw::ExecutionContext{}),
                                configs,
                                options,
                                min_scored,
                                prior ? &*prior : nullptr);
      },
      "model"_a,
      "problem"_a,
      "hardware"_a,
      "configs"_a,
      "min_scored"_a = std::size_t{0},
      "context"_a    = nb::none(),
      "tie_break"_a  = tw::TieBreak::PoolOrder,
      "prior"_a      = nb::none(),
      "launch"_a     = nb::none(),
      "Same result as CandidateSet(model, configs, tie_break, launch).rank(problem, hardware, "
      "min_scored, context, prior), with the same ValueErrors.");
}
