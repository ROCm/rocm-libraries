// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
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

  nb::class_<tw::ModelInfo>(m, "ModelInfo")
      .def_ro("arch", &tw::ModelInfo::arch)
      .def_ro("feature_catalog_hash", &tw::ModelInfo::feature_catalog_hash)
      .def_ro("weight_type", &tw::ModelInfo::weight_type)
      .def_ro("n_cells", &tw::ModelInfo::n_cells)
      .def_ro("n_splits", &tw::ModelInfo::n_splits)
      .def("__repr__", [](const tw::ModelInfo& i) {
        return "ModelInfo(arch='" + i.arch + "', feature_catalog_hash='" + i.feature_catalog_hash +
               "', weight_type=" + std::to_string(static_cast<int>(i.weight_type)) +
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
             tw::TieBreak tie_break) {
            model_of(model);
            new (self) tw::CandidateSet(model.ptr, std::move(configs), tw::PoolOptions{tie_break});
          },
          "model"_a,
          "configs"_a,
          "tie_break"_a = tw::TieBreak::PoolOrder,
          "Raises ValueError for a config with an empty or duplicate attribute name.")
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
          "configs in input order. context None means exclusive use of the device; every "
          "config is unscored when the model does not support the context. With "
          "TieBreak.Prior, prior (one float per config) orders equal scores; ValueError "
          "when it has the wrong length.")
      .def_prop_ro("configs", [](const tw::CandidateSet& s) { return s.configs(); })
      .def("__len__", [](const tw::CandidateSet& s) { return s.configs().size(); });

  m.def("feature_catalog_hash", []() { return std::string(tw::feature_catalog_hash()); });

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
      [](const std::string& logic_stem, const std::string& dir) -> nb::object {
        std::string error;
        tw::ModelPtr model;
        {
          nb::gil_scoped_release release;
          model = tw::load_model_by_index(logic_stem, dir, &error);
        }
        if (model) return nb::cast(PyModel{std::move(model)});
        if (!error.empty()) throw std::invalid_argument(error);
        return nb::none();
      },
      "logic_stem"_a,
      "dir"_a,
      "Load the model <dir>/tilewright_index lists for logic_stem. Returns None when the index "
      "or the stem is absent; raises ValueError when the listed model fails to load.");

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
      [](const PyModel& model, const tw::Problem& p, const tw::Config& c, const tw::Hardware& h) {
        return tw::compute_features(model_of(model), p, c, h);
      },
      "model"_a,
      "problem"_a,
      "config"_a,
      "hardware"_a,
      "Raw (unwhitened) query, item and interaction features; empty for invalid hardware.");

  m.def(
      "attribute_names",
      [](const PyModel& model) { return tw::attribute_names(model_of(model)); },
      "model"_a,
      "Names of the Config attributes the model's feature catalog reads (v2 models: none).");

  m.def(
      "supports",
      [](const PyModel& model, const tw::ExecutionContext& context, const tw::Hardware& h) {
        return tw::supports(model_of(model), context, h);
      },
      "model"_a,
      "context"_a,
      "hardware"_a,
      "True when the model was trained for the context on this hardware (v2 models: every CU "
      "and Schedule.Default).");

  m.def(
      "rank_configs",
      [](const PyModel& model,
         tw::Problem p,
         tw::Hardware h,
         const std::vector<tw::Config>& configs,
         std::size_t min_scored,
         std::optional<tw::ExecutionContext> context,
         tw::TieBreak tie_break,
         std::optional<std::vector<double>> prior) {
        const tw::Model& mdl = model_of(model);
        nb::gil_scoped_release release;
        return tw::rank_configs(mdl,
                                p,
                                h,
                                context.value_or(tw::ExecutionContext{}),
                                configs,
                                min_scored,
                                tie_break,
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
      "Same result as CandidateSet(model, configs, tie_break).rank(problem, hardware, "
      "min_scored, context, prior).");
}
