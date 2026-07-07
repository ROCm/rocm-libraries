/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2026 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "origami/comm/tensor.hpp"

namespace origami::comm {

double wire_factor(primitive_t op, int world_size) {
  const double n = static_cast<double>(world_size);
  // Standard bus-bandwidth factors (see header for the derivation of each). Kept
  // as a closed switch so adding a primitive without a factor is a compile error.
  switch (op) {
    case primitive_t::all_reduce: return 2.0 * (n - 1.0) / n;
    case primitive_t::all_gather: return (n - 1.0);
    case primitive_t::reduce_scatter: return (n - 1.0);
    case primitive_t::broadcast: return 1.0;
    case primitive_t::all_to_all: return (n - 1.0) / n;
  }
  // Every enumerator is handled above; reaching here means a corrupted/out-of-range
  // value rather than a genuinely "unknown" collective.
  throw std::invalid_argument(std::string{"unknown collective: "} +
                              std::string{primitive_name(op)});
}

std::size_t msg_bytes_for_predict_row(primitive_t op, std::size_t per_rank_bytes, int world_size) {
  // reduce_scatter is the lone op whose benchmark msg_bytes is the *aggregate*
  // pre-scatter buffer (per_rank × N). predict_row re-divides msg_bytes by N
  // internally, so scaling up here makes that division cancel and the two agree.
  // Every other op already reports its per-rank buffer.
  if (op == primitive_t::reduce_scatter) {
    return per_rank_bytes * static_cast<std::size_t>(world_size);
  }
  return per_rank_bytes;
}

data_type_t normalize_dtype(std::string_view dt) {
  auto strip = [](std::string s) {
    // Callers commonly pass framework-qualified, mixed-case names like
    // "torch.float32" or "np.float16". Lowercasing and dropping the library
    // prefix lets the alias table below stay a flat list of bare names.
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    auto erase_prefix = [&](std::string_view p) {
      if (s.rfind(p, 0) == 0) s.erase(0, p.size());
    };
    erase_prefix("torch.");
    erase_prefix("np.");
    erase_prefix("numpy.");
    return s;
  };
  const std::string key = strip(std::string{dt});
  if (key == "bf16" || key == "bfloat16") return data_type_t::BFloat16;
  if (key == "fp16" || key == "float16" || key == "half") return data_type_t::Half;
  if (key == "fp32" || key == "float32" || key == "float") return data_type_t::Float;
  if (key == "fp64" || key == "float64" || key == "double") return data_type_t::Double;
  if (key == "fp8") return data_type_t::Float8;
  if (key == "int8") return data_type_t::Int8;
  throw std::invalid_argument(std::string{"unsupported dtype: "} + std::string{dt});
}

data_type_t normalize_dtype(data_type_t dt) noexcept { return dt; }

full_mn_t per_rank_shape_to_full_mn(const std::vector<std::size_t>& shape,
                                    int dim,
                                    int world_size) {
  if (shape.empty()) {
    throw std::invalid_argument("input_shape must have at least one dimension");
  }
  for (auto d : shape) {
    if (d == 0) { throw std::invalid_argument("input_shape has non-positive entries"); }
  }
  if (world_size < 1) { throw std::invalid_argument("world_size must be >= 1"); }

  const int rank_ndim = static_cast<int>(shape.size());
  // Negative dim indexes from the end, Python/NumPy style (-1 == last axis).
  const int norm_dim = (dim >= 0) ? dim : dim + rank_ndim;
  if (norm_dim < 0 || norm_dim >= rank_ndim) {
    throw std::invalid_argument("dim out of range for input_shape");
  }

  // The cost model is strictly 2D [M, N], so an arbitrary-rank tensor is folded
  // into a matrix: every leading dim collapses into M (rows) and the last dim is
  // kept as N (columns). This matches row-major memory order, so the collapse is
  // contiguity-preserving.
  const std::size_t n_per_rank = shape.back();
  std::size_t m_per_rank       = 1;
  for (int i = 0; i + 1 < rank_ndim; ++i) m_per_rank *= shape[i];

  // Only the sharded axis is partial on each rank; scale it by world_size to
  // recover the full logical extent. The returned split_dim (0=rows, 1=cols)
  // lets the lower layers re-derive a rank's tile by the inverse division.
  const int last_dim = rank_ndim - 1;
  if (norm_dim == last_dim) {
    // Sharded on the last axis → the columns (N) were split across ranks.
    return {m_per_rank, n_per_rank * static_cast<std::size_t>(world_size), 1};
  }
  // Sharded on any leading axis → the rows (M) were split across ranks.
  return {m_per_rank * static_cast<std::size_t>(world_size), n_per_rank, 0};
}

tensor_collective_prediction_t predict_tensor_collective(
    std::string_view op,
    const std::vector<std::size_t>& input_shape,
    data_type_t dtype,
    int world_size,
    const system_t& system,
    int dim,
    int nchannels,
    std::string_view framework,
    const heuristics_t& heur) {
  // Parse the op name to its enum once, here at the public boundary, so the rest
  // of the pipeline works in terms of primitive_t rather than re-comparing strings.
  const primitive_t prim = primitive_from_name(op);
  if (world_size < 1) { throw std::invalid_argument("world_size must be >= 1"); }

  // Buffer each rank actually holds, in bytes (shared by both paths).
  std::size_t per_rank_elements = 1;
  for (auto d : input_shape) per_rank_elements *= d;
  const std::size_t per_rank_bytes =
      per_rank_elements * static_cast<std::size_t>(dtype_bytes(dtype));

  // Host-side launch floor; paid on every path, including the W=1 no-op.
  const double overhead_us = heur.framework_overhead_us(framework);

  // Quantities that exist only once bytes actually cross the fabric. The defaults
  // below ARE the single-rank (W=1) answer: a collective on one rank is a local
  // no-op, so nothing is sent (wire/backend = 0), msg_bytes is just the buffer,
  // and the tile is the whole per-rank tensor collapsed to 2D (leading dims into
  // rows, last dim into columns; nothing to un-shard). The framework still
  // charged its overhead, hence the early defaults rather than an early return.
  double backend_us               = 0.0;
  std::size_t wire_bytes_per_rank = 0;
  std::size_t msg_bytes           = per_rank_bytes;
  tile_shape_t gpu_tile;
  if (!input_shape.empty()) {
    std::size_t outer = 1;
    for (std::size_t i = 0; i + 1 < input_shape.size(); ++i) outer *= input_shape[i];
    gpu_tile = tile_shape_t{outer, input_shape.back(), dtype, /*split_dim=*/0, /*contiguous=*/true};
  }

  if (world_size > 1) {
    // Reporting-only cross-check: fabric bytes per rank implied by the textbook
    // bus-bandwidth factor. The prediction comes from the algorithm model
    // (predict_row), not from this number.
    wire_bytes_per_rank = static_cast<std::size_t>(wire_factor(prim, world_size) *
                                                   static_cast<double>(per_rank_bytes));

    // Lower the per-rank tensor to the byte-level model: reconstruct the full
    // [M,N] extent + split axis, and translate the per-rank byte count into
    // predict_row's msg_bytes convention (which differs only for reduce_scatter).
    const auto full = per_rank_shape_to_full_mn(input_shape, dim, world_size);
    msg_bytes       = msg_bytes_for_predict_row(prim, per_rank_bytes, world_size);
    backend_us      = predict_row(op,
                                  msg_bytes,
                                  world_size,
                                  nchannels,
                                  system,
                                  full.M_full,
                                  full.N_full,
                                  full.split_dim,
                                  heur);

    // Reconstruct one rank's tile: divide the full extent along the sharded axis
    // by world_size (the inverse of the multiply per_rank_shape_to_full_mn
    // applied), leaving the unsharded axis at full size.
    gpu_tile = tile_shape_t{
        (full.split_dim == 0) ? full.M_full / static_cast<std::size_t>(world_size) : full.M_full,
        (full.split_dim == 1) ? full.N_full / static_cast<std::size_t>(world_size) : full.N_full,
        dtype,
        full.split_dim,
        /*contiguous=*/true};
  }

  // predict_row returns GPU/backend time; the caller-visible latency adds the
  // host framework floor on top (backend_us is 0 for the W=1 no-op).
  tensor_collective_prediction_t out{};
  out.predicted_us          = backend_us + overhead_us;
  out.op                    = std::string{op};
  out.input_shape           = input_shape;
  out.dim                   = dim;
  out.world_size            = world_size;
  out.nchannels             = nchannels;
  out.dtype                 = dtype;
  out.per_rank_bytes        = per_rank_bytes;
  out.wire_bytes_per_rank   = wire_bytes_per_rank;
  out.msg_bytes             = msg_bytes;
  out.gpu_tile              = gpu_tile;
  out.framework             = std::string{framework};
  out.framework_overhead_us = overhead_us;
  return out;
}

tensor_collective_prediction_t predict_tensor_collective(
    std::string_view op,
    const std::vector<std::size_t>& input_shape,
    std::string_view dtype_name,
    int world_size,
    const system_t& system,
    int dim,
    int nchannels,
    std::string_view framework,
    const heuristics_t& heur) {
  return predict_tensor_collective(op,
                                   input_shape,
                                   normalize_dtype(dtype_name),
                                   world_size,
                                   system,
                                   dim,
                                   nchannels,
                                   framework,
                                   heur);
}

}  // namespace origami::comm
