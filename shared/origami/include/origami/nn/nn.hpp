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

#if !ORIGAMI_ENABLE_NN
#  error "origami::nn requires ORIGAMI_ENABLE_NN=ON"
#endif

#include "origami/nn/types.hpp"
#include "origami/origami_export.h"

#include <string>

namespace origami::nn {

ORIGAMI_EXPORT model_handle_t load_model(const std::string& path);

ORIGAMI_EXPORT model_handle_t load_model_by_index(const std::string& logic_stem,
                                                  backend_id_t backend,
                                                  const std::string& hint_dir = "");

ORIGAMI_EXPORT library_models_t load_models_for_logic(const std::string& logic_stem,
                                                      const std::string& hint_dir = "");

ORIGAMI_EXPORT void unload_model(model_handle_t handle);

ORIGAMI_EXPORT const model_info_t* model_info(model_handle_t handle);

ORIGAMI_EXPORT void set_default_model(model_handle_t handle);

ORIGAMI_EXPORT model_handle_t default_model(backend_id_t backend);

}  // namespace origami::nn
