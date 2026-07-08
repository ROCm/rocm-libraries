/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2025-2026 AMD ROCm(TM) Software
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

#pragma once

#include <cstddef>
#include <cstdint>

namespace origami::nn {

using model_handle_t = int;

constexpr model_handle_t invalid_handle = -1;

enum class backend_id_t : std::uint8_t {
  tilewright_v1,
  embedding_similarity_v1,
};

/// Backend-specific knobs passed through rank_options_t::nn.
struct inference_options_t {
  std::size_t min_scored            = 0;
  int         force_cell            = -1;
  bool        use_smart_k_whitelist = true;
};

/// Per-library handles for both ML backends (either may be invalid_handle).
struct library_models_t {
  model_handle_t tilewright           = invalid_handle;
  model_handle_t embedding_similarity = invalid_handle;
};

}  // namespace origami::nn
