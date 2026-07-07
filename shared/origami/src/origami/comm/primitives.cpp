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

#include "origami/comm/primitives.hpp"

namespace origami::comm {

resolved_work_t resolve_work_graph(const std::vector<op_t>& ops, const iter_dims_t& iter) noexcept {
  resolved_work_t out{};
  for (const op_t& op : ops) {
    std::visit(
        [&](const auto& concrete) {
          using T                        = std::decay_t<decltype(concrete)>;
          const functional_unit_work_t w = concrete.resolve(iter);
          if constexpr (std::is_same_v<T, signal_t> || std::is_same_v<T, wait_t>) {
            out.sync_work += w;
          } else {
            out.iter_work += w;
          }
        },
        op);
  }
  return out;
}

}  // namespace origami::comm
