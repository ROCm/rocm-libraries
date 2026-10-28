// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Library stamp of Origami Boost models: the structural-field table, the canonical manifest of
// a library's PredictionMatching rows, the diff that decides whether a model may run on a
// library other than the one it was trained on, and the MANI section text. The exporter
// (origami_boost_manifest.py) mirrors this file; its check CLI must reach the same decision.

#include <algorithm>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "origami/nn/boost/library_stamp.hpp"
#include "origami/nn/boost/load_limits.hpp"
#include "origami/nn/boost/loaded_model.hpp"
#include "origami/nn/boost/sha256.hpp"

namespace origami::nn::boost {

namespace {

std::string b(bool v) { return v ? "1" : "0"; }
std::string n(long long v) { return std::to_string(v); }
std::string n(std::size_t v) { return std::to_string(v); }

using value_fn = std::string (*)(const config_t&, const tensile_params_t&, const kernel_fields_t&);

struct field_def_t {
  structural_field_t field;
  value_fn           value;
};

// The structural fields. gate = true: an added kernel with a value no training kernel had
// refuses the model (the families whose absence from training costs most of the gain: DTL,
// MFMA type and non-MFMA, LDS mode (TLDS), one LDS buffer, MT_K). The rest are recorded, and
// reported when new, but do not refuse. Values are the config_t hipBLASLt builds for
// rank_configs (Serialization/PredictionLibrary.hpp), plus the solution's NumLoadsCoalesced.
const field_def_t kFields[] = {
    {{"dtl", true},
     [](const config_t&, const tensile_params_t& t, const kernel_fields_t&) {
       return b(t.direct_to_lds_a) + b(t.direct_to_lds_b);
     }},
    {{"mfma", true},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) {
       // hipBLASLt gives non-MFMA (dot2 / MAC) kernels the instruction 1x1x64.
       return b(!(c.mi.m == 1 && c.mi.n == 1 && c.mi.k == 64));
     }},
    {{"mi", true},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) {
       return n(c.mi.m) + "x" + n(c.mi.n) + "x" + n(c.mi.k);
     }},
    {{"lds_mode", true},
     [](const config_t&, const tensile_params_t& t, const kernel_fields_t&) {
       return n(static_cast<long long>(t.unroll_major_lds_a)) + "," +
              n(static_cast<long long>(t.unroll_major_lds_b)) + "," + b(t.lds_transpose_read_a) +
              "," + b(t.lds_transpose_read_b);
     }},
    {{"one_lds_buffer", true},
     [](const config_t&, const tensile_params_t& t, const kernel_fields_t&) {
       return b(t.one_lds_buffer);
     }},
    {{"mt_k", true},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) { return n(c.mt.k); }},
    {{"mt_m", false},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) { return n(c.mt.m); }},
    {{"mt_n", false},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) { return n(c.mt.n); }},
    {{"stream_k", false},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) {
       return n(static_cast<long long>(c.stream_k));
     }},
    {{"pgr", false},
     [](const config_t&, const tensile_params_t& t, const kernel_fields_t&) {
       return n(static_cast<long long>(t.prefetch_global_read));
     }},
    {{"lsu", false},
     [](const config_t&, const tensile_params_t& t, const kernel_fields_t&) {
       return n(static_cast<long long>(t.local_split_u));
     }},
    {{"wave_group", false},
     [](const config_t&, const tensile_params_t& t, const kernel_fields_t&) {
       return n(static_cast<long long>(t.wave_group_m)) + "x" +
              n(static_cast<long long>(t.wave_group_n));
     }},
    {{"wavefront_size", false},
     [](const config_t&, const tensile_params_t& t, const kernel_fields_t&) {
       return n(static_cast<long long>(t.wavefront_size));
     }},
    {{"source_swap", false},
     [](const config_t&, const tensile_params_t& t, const kernel_fields_t&) {
       return b(t.source_swap);
     }},
    {{"cms", false},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) {
       return b(c.hand_optimized_main_loop);
     }},
    {{"subtile", false},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) { return b(c.subtile); }},
    {{"dtv", false},
     [](const config_t&, const tensile_params_t& t, const kernel_fields_t&) {
       return b(t.direct_to_vgpr_a) + b(t.direct_to_vgpr_b);
     }},
    {{"cache_hints", false},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) {
       return n(static_cast<long long>(c.cache_hints_a)) + "," +
              n(static_cast<long long>(c.cache_hints_b)) + "," +
              n(static_cast<long long>(c.cache_hints_d));
     }},
    {{"vector_width", false},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) {
       return n(static_cast<long long>(c.vector_width_a)) + "," +
              n(static_cast<long long>(c.vector_width_b));
     }},
    {{"grvw", false},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) {
       return n(c.grvw_a) + "," + n(c.grvw_b);
     }},
    {{"gwvw_d", false},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) { return n(c.gwvw_d); }},
    {{"occupancy", false},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) {
       return n(static_cast<long long>(c.occupancy));
     }},
    {{"wgm", false},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) {
       return n(static_cast<long long>(c.workgroup_mapping));
     }},
    {{"cluster_dim", false},
     [](const config_t& c, const tensile_params_t&, const kernel_fields_t&) {
       return n(c.cluster_dim.m) + "," + n(c.cluster_dim.n) + "," + n(c.cluster_dim.k);
     }},
    {{"nlc", false},
     [](const config_t&, const tensile_params_t&, const kernel_fields_t& f) {
       return n(static_cast<long long>(f.num_loads_coalesced_a)) + "," +
              n(static_cast<long long>(f.num_loads_coalesced_b));
     }},
};

constexpr std::size_t kNumFields = sizeof(kFields) / sizeof(kFields[0]);

std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> out;
  std::size_t              start = 0;
  for (;;) {
    const std::size_t tab = line.find('\t', start);
    out.push_back(line.substr(start, tab == std::string::npos ? std::string::npos : tab - start));
    if (tab == std::string::npos) return out;
    start = tab + 1;
  }
}

std::string set_text(const std::map<std::string, std::set<std::string>>& m) {
  std::string out;
  for (const auto& f : structural_fields()) {
    const auto it = m.find(f.name);
    if (it == m.end()) continue;
    std::string vals;
    for (const auto& v : it->second) vals += (vals.empty() ? "" : ",") + v;
    out += (out.empty() ? "" : "; ") + std::string(f.name) + " {" + vals + "}";
  }
  return out;
}

std::string json_str(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

std::string json_sets(const std::map<std::string, std::set<std::string>>& m) {
  std::string out = "{";
  bool        first = true;
  for (const auto& f : structural_fields()) {
    const auto it = m.find(f.name);
    if (it == m.end()) continue;
    out += (first ? "" : ", ") + json_str(f.name) + ": [";
    first = false;
    bool first_v = true;
    for (const auto& v : it->second) {
      out += (first_v ? "" : ", ") + json_str(v);
      first_v = false;
    }
    out += "]";
  }
  return out + "}";
}

}  // namespace

const std::vector<structural_field_t>& structural_fields() {
  static const std::vector<structural_field_t> all = [] {
    std::vector<structural_field_t> v;
    for (const auto& d : kFields) v.push_back(d.field);
    return v;
  }();
  return all;
}

std::vector<std::string> structural_values(const config_t& config, const kernel_fields_t& fields) {
  static const tensile_params_t defaults{};
  const tensile_params_t& t = config.has_tensile_params() ? config.tensile() : defaults;
  std::vector<std::string> v;
  v.reserve(kNumFields);
  for (const auto& d : kFields) v.push_back(d.value(config, t, fields));
  return v;
}

std::string name_digest(const std::string& name) {
  return sha256_hex(name.data(), name.size()).substr(0, 16);
}

std::vector<manifest_entry_t> canonical_manifest(const std::vector<library_kernel_t>& kernels) {
  std::vector<manifest_entry_t> all;
  all.reserve(kernels.size());
  for (const auto& k : kernels)
    all.push_back({name_digest(k.name), k.index, structural_values(k.config, k.fields), k.name});
  std::sort(all.begin(), all.end(), [](const manifest_entry_t& a, const manifest_entry_t& b) {
    return std::tie(a.digest, a.values, a.index) < std::tie(b.digest, b.values, b.index);
  });
  std::vector<manifest_entry_t> out;
  for (auto& e : all)
    if (out.empty() || out.back().digest != e.digest || out.back().values != e.values)
      out.push_back(std::move(e));
  return out;
}

const char* to_string(library_decision_t decision) {
  switch (decision) {
    case library_decision_t::accept: return "ACCEPT";
    case library_decision_t::accept_with_additions: return "ACCEPT-WITH-ADDITIONS";
    case library_decision_t::retrain_required: return "RETRAIN-REQUIRED";
  }
  return "?";
}

library_diff_t diff_library(const ensemble_t& model, const std::vector<manifest_entry_t>& library) {
  library_diff_t d;
  d.dtype = model.head_string("dtype");

  using key_t = std::pair<std::string, std::vector<std::string>>;
  std::set<key_t> trained;
  for (const auto& e : model.library_manifest()) trained.insert({e.digest, e.values});
  std::set<key_t> present;
  std::set<std::string> affected;
  for (const auto& e : library) {
    if (e.values.size() != kNumFields || !present.insert({e.digest, e.values}).second) continue;
    if (trained.count({e.digest, e.values})) {
      ++d.kept;
      continue;
    }
    ++d.added;
    for (std::size_t f = 0; f < kNumFields; ++f) {
      if (model.seen_values()[f].count(e.values[f])) continue;
      if (kFields[f].field.gate) {
        d.new_values[kFields[f].field.name].insert(e.values[f]);
        affected.insert(e.name.empty() ? e.digest : e.name);
      } else {
        d.new_ungated_values[kFields[f].field.name].insert(e.values[f]);
      }
    }
  }
  d.model_kernels   = trained.size();
  d.library_kernels = present.size();
  d.removed         = d.model_kernels - d.kept;
  d.affected.assign(affected.begin(), affected.end());
  d.decision = !d.new_values.empty() ? library_decision_t::retrain_required
               : d.added > 0        ? library_decision_t::accept_with_additions
                                    : library_decision_t::accept;
  return d;
}

std::string library_diff_t::summary() const {
  std::string s = to_string(decision);
  s += ": " + std::to_string(kept) + " of " + std::to_string(model_kernels) +
       " training kernels present, " + std::to_string(removed) + " removed, " +
       std::to_string(added) + " added";
  if (decision == library_decision_t::retrain_required)
    s += "; " + std::to_string(affected.size()) +
         " added kernels carry structural values never seen in training: " + set_text(new_values);
  if (!new_ungated_values.empty()) s += "; new ungated values: " + set_text(new_ungated_values);
  return s;
}

std::string library_diff_t::json() const {
  std::string affected_json = "[";
  for (std::size_t i = 0; i < affected.size(); ++i)
    affected_json += (i ? ", " : "") + json_str(affected[i]);
  affected_json += "]";
  return std::string("{\"decision\": ") + json_str(to_string(decision)) +
         ", \"dtype\": " + json_str(dtype) + ", \"model_kernels\": " + std::to_string(model_kernels) +
         ", \"library_kernels\": " + std::to_string(library_kernels) +
         ", \"kept\": " + std::to_string(kept) + ", \"removed\": " + std::to_string(removed) +
         ", \"added\": " + std::to_string(added) +
         ", \"new_structural_values\": " + json_sets(new_values) +
         ", \"new_ungated_values\": " + json_sets(new_ungated_values) +
         ", \"affected_kernels\": " + affected_json + ", \"summary\": " + json_str(summary()) + "}";
}

std::string manifest_section_text(const std::string&                       dtype,
                                  const std::vector<manifest_entry_t>&     entries,
                                  const std::vector<std::set<std::string>>& seen) {
  std::ostringstream o;
  o << "manifest\t" << detail::kManifestVersion << "\n"
    << "dtype\t" << dtype << "\n";
  for (const auto& d : kFields) o << "field\t" << d.field.name << "\t" << (d.field.gate ? 1 : 0) << "\n";
  for (std::size_t f = 0; f < kNumFields; ++f) {
    o << "seen\t" << kFields[f].field.name;
    if (f < seen.size())
      for (const auto& v : seen[f]) o << "\t" << v;
    o << "\n";
  }
  for (const auto& e : entries) {
    o << "kernel\t" << e.index << "\t" << e.digest;
    for (const auto& v : e.values) o << "\t" << v;
    o << "\n";
  }
  return o.str();
}

std::vector<std::set<std::string>> seen_structural_values(const std::vector<manifest_entry_t>& entries) {
  std::vector<std::set<std::string>> seen(kNumFields);
  for (const auto& e : entries)
    for (std::size_t f = 0; f < kNumFields && f < e.values.size(); ++f) seen[f].insert(e.values[f]);
  return seen;
}

namespace detail {

manifest_parse_t parse_manifest_section(const std::string&                  text,
                                        std::string&                        dtype,
                                        std::vector<manifest_entry_t>&      entries,
                                        std::vector<std::set<std::string>>& seen,
                                        std::string&                        error) {
  std::istringstream in(text);
  std::string        line;
  std::size_t        field_no = 0, seen_no = 0;
  bool               header = false;
  entries.clear();
  seen.assign(kNumFields, {});
  while (std::getline(in, line)) {
    const auto t = split_tabs(line);
    if (!header) {
      if (t.size() != 2 || t[0] != "manifest" || t[1] != detail::kManifestVersion) {
        error = "MANI: unknown manifest version";
        return manifest_parse_t::unsupported;
      }
      header = true;
    } else if (t[0] == "dtype" && t.size() == 2) {
      dtype = t[1];
    } else if (t[0] == "field" && t.size() == 3) {
      if (field_no >= kNumFields || t[1] != kFields[field_no].field.name ||
          t[2] != (kFields[field_no].field.gate ? "1" : "0")) {
        error = "MANI: structural field table differs from this runtime's (field " + t[1] + ")";
        return manifest_parse_t::unsupported;
      }
      ++field_no;
    } else if (t[0] == "seen" && t.size() >= 2) {
      if (seen_no >= kNumFields || t[1] != kFields[seen_no].field.name) {
        error = "MANI: seen values out of order";
        return manifest_parse_t::malformed;
      }
      seen[seen_no].insert(t.begin() + 2, t.end());
      ++seen_no;
    } else if (t[0] == "kernel" && t.size() == 3 + kNumFields) {
      manifest_entry_t e;
      try {
        std::size_t used = 0;
        e.index          = std::stoll(t[1], &used);
        if (used != t[1].size()) throw std::invalid_argument("index");
      } catch (const std::exception&) {
        error = "MANI: bad kernel index " + t[1];
        return manifest_parse_t::malformed;
      }
      if (t[2].size() != 16 || t[2].find_first_not_of("0123456789abcdef") != std::string::npos) {
        error = "MANI: bad kernel digest " + t[2];
        return manifest_parse_t::malformed;
      }
      e.digest = t[2];
      e.values.assign(t.begin() + 3, t.end());
      entries.push_back(std::move(e));
    } else {
      error = "MANI: malformed line: " + line.substr(0, 80);
      return manifest_parse_t::malformed;
    }
  }
  if (!header || field_no != kNumFields || seen_no != kNumFields || entries.empty() || dtype.empty()) {
    error = field_no != kNumFields && header
                ? "MANI: structural field table differs from this runtime's (field count)"
                : "MANI: incomplete manifest";
    return field_no != kNumFields && header ? manifest_parse_t::unsupported
                                            : manifest_parse_t::malformed;
  }
  return manifest_parse_t::ok;
}

}  // namespace detail
}  // namespace origami::nn::boost
