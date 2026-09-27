// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <execution>
#include <functional>
#include <vector>

int main()
{
  using namespace std;

  vector<int> v{};

  adjacent_difference(execution::par_unseq, cbegin(v), cend(v), begin(v));
  adjacent_difference(execution::par_unseq, cbegin(v), cend(v), begin(v), minus<>{});
  adjacent_find(execution::par_unseq, cbegin(v), cend(v));
  adjacent_find(execution::par_unseq, cbegin(v), cend(v), equal_to<>{});
  all_of(execution::par_unseq, cbegin(v), cend(v), logical_not<>{});
  copy(execution::par_unseq, cbegin(v), cend(v), begin(v));
  copy_if(execution::par_unseq, cbegin(v), cend(v), begin(v), logical_not<>{});
  copy_n(execution::par_unseq, cbegin(v), size(v), begin(v));
  count(execution::par_unseq, cbegin(v), cend(v), 42);
  count_if(execution::par_unseq, cbegin(v), cend(v), logical_not<>{});
  destroy(execution::par_unseq, begin(v), end(v));
  destroy_n(execution::par_unseq, begin(v), size(v));
  equal(execution::par_unseq, cbegin(v), cend(v), cbegin(v));
  equal(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v));
  equal(execution::par_unseq, cbegin(v), cend(v), cbegin(v), equal_to<>{});
  equal(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), equal_to<>{});
  exclusive_scan(execution::par_unseq, cbegin(v), cend(v), begin(v), 0);
  exclusive_scan(execution::par_unseq, cbegin(v), cend(v), begin(v), 0, plus<>{});
  fill(execution::par_unseq, begin(v), end(v), 0);
  fill_n(execution::par_unseq, begin(v), size(v), 0);
  find(execution::par_unseq, cbegin(v), cend(v), 42);
  find_end(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v));
  find_end(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), equal_to<>{});
  find_first_of(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v));
  find_first_of(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), equal_to<>{});
  find_if(execution::par_unseq, cbegin(v), cend(v), logical_not<>{});
  find_if_not(execution::par_unseq, cbegin(v), cend(v), logical_not<>{});
  for_each(execution::par_unseq, cbegin(v), cend(v), [](auto&&) {});
  for_each_n(execution::par_unseq, cbegin(v), size(v), [](auto&&) {});
  generate(execution::par_unseq, begin(v), end(v), []() {
    return 42;
  });
  generate_n(execution::par_unseq, begin(v), size(v), []() {
    return 42;
  });
  includes(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v));
  includes(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), equal_to<>{});
  inclusive_scan(execution::par_unseq, cbegin(v), cend(v), begin(v));
  inclusive_scan(execution::par_unseq, cbegin(v), cend(v), begin(v), plus<>{});
  inclusive_scan(execution::par_unseq, cbegin(v), cend(v), begin(v), plus<>{}, 0);
  inplace_merge(execution::par_unseq, begin(v), begin(v), end(v));
  inplace_merge(execution::par_unseq, begin(v), begin(v), end(v), less<>{});
  is_heap(execution::par_unseq, cbegin(v), cend(v));
  is_heap(execution::par_unseq, cbegin(v), cend(v), less<>{});
  is_heap_until(execution::par_unseq, cbegin(v), cend(v));
  is_heap_until(execution::par_unseq, cbegin(v), cend(v), less<>{});
  is_partitioned(execution::par_unseq, cbegin(v), cend(v), logical_not<>{});
  is_sorted(execution::par_unseq, cbegin(v), cend(v));
  is_sorted(execution::par_unseq, cbegin(v), cend(v), less<>{});
  is_sorted_until(execution::par_unseq, cbegin(v), cend(v));
  is_sorted_until(execution::par_unseq, cbegin(v), cend(v), less<>{});
  lexicographical_compare(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v));
  lexicographical_compare(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), less<>{});
  max_element(execution::par_unseq, cbegin(v), cend(v));
  max_element(execution::par_unseq, cbegin(v), cend(v), less<>{});
  merge(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), begin(v));
  merge(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), begin(v), less<>{});
  min_element(execution::par_unseq, cbegin(v), cend(v));
  min_element(execution::par_unseq, cbegin(v), cend(v), less<>{});
  minmax_element(execution::par_unseq, cbegin(v), cend(v));
  minmax_element(execution::par_unseq, cbegin(v), cend(v), less<>{});
  mismatch(execution::par_unseq, cbegin(v), cend(v), cbegin(v));
  mismatch(execution::par_unseq, cbegin(v), cend(v), cbegin(v), equal_to<>{});
  mismatch(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v));
  mismatch(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), equal_to<>{});
  move(execution::par_unseq, cbegin(v), cend(v), begin(v));
  none_of(execution::par_unseq, cbegin(v), cend(v), logical_not<>{});
  nth_element(execution::par_unseq, begin(v), begin(v), end(v));
  nth_element(execution::par_unseq, begin(v), begin(v), end(v), less<>{});
  partial_sort(execution::par_unseq, begin(v), begin(v), end(v));
  partial_sort(execution::par_unseq, begin(v), begin(v), end(v), less<>{});
  partial_sort_copy(execution::par_unseq, cbegin(v), cend(v), begin(v), end(v));
  partial_sort_copy(execution::par_unseq, cbegin(v), cend(v), begin(v), end(v), less<>{});
  partition(execution::par_unseq, begin(v), end(v), logical_not<>{});
  partition_copy(execution::par_unseq, cbegin(v), cend(v), begin(v), begin(v), logical_not<>{});
  reduce(execution::par_unseq, cbegin(v), cend(v));
  reduce(execution::par_unseq, cbegin(v), cend(v), 0);
  reduce(execution::par_unseq, cbegin(v), cend(v), 0, plus<>{});
  remove(execution::par_unseq, begin(v), end(v), 42);
  remove_copy(execution::par_unseq, cbegin(v), cend(v), begin(v), 42);
  remove_copy_if(execution::par_unseq, cbegin(v), cend(v), begin(v), logical_not<>{});
  remove_if(execution::par_unseq, begin(v), end(v), logical_not<>{});
  replace(execution::par_unseq, begin(v), end(v), 42, 69);
  replace_copy(execution::par_unseq, cbegin(v), cend(v), begin(v), 42, 69);
  replace_copy_if(execution::par_unseq, cbegin(v), cend(v), begin(v), logical_not<>{}, 42);
  replace_if(execution::par_unseq, begin(v), end(v), logical_not<>{}, 42);
  reverse(execution::par_unseq, begin(v), end(v));
  reverse_copy(execution::par_unseq, cbegin(v), cend(v), begin(v));
  rotate(execution::par_unseq, begin(v), begin(v), end(v));
  rotate_copy(execution::par_unseq, cbegin(v), cbegin(v), cend(v), begin(v));
  search(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v));
  search(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), equal_to<>{});
  search_n(execution::par_unseq, cbegin(v), cend(v), size(v), 42);
  search_n(execution::par_unseq, cbegin(v), cend(v), size(v), 42, equal_to<>{});
  set_difference(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), begin(v));
  set_difference(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), begin(v), equal_to<>{});
  set_intersection(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), begin(v));
  set_intersection(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), begin(v), equal_to<>{});
  set_symmetric_difference(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), begin(v));
  set_symmetric_difference(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), begin(v), equal_to<>{});
  set_union(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), begin(v));
  set_union(execution::par_unseq, cbegin(v), cend(v), cbegin(v), cend(v), begin(v), equal_to<>{});
  sort(execution::par_unseq, begin(v), end(v));
  sort(execution::par_unseq, begin(v), end(v), less<>{});
  stable_partition(execution::par_unseq, begin(v), end(v), logical_not<>{});
  stable_sort(execution::par_unseq, begin(v), end(v));
  stable_sort(execution::par_unseq, begin(v), end(v), less<>{});
  swap_ranges(execution::par_unseq, begin(v), end(v), begin(v));
  transform(execution::par_unseq, cbegin(v), cend(v), begin(v), logical_not<>{});
  transform(execution::par_unseq, cbegin(v), cend(v), cbegin(v), begin(v), plus<>{});
  transform_exclusive_scan(execution::par_unseq, cbegin(v), cend(v), begin(v), 0, plus<>{}, logical_not<>{});
  transform_inclusive_scan(execution::par_unseq, cbegin(v), cend(v), begin(v), plus<>{}, logical_not<>{});
  transform_inclusive_scan(execution::par_unseq, cbegin(v), cend(v), begin(v), plus<>{}, logical_not<>{}, 42);
  transform_reduce(execution::par_unseq, cbegin(v), cend(v), cbegin(v), 42);
  transform_reduce(execution::par_unseq, cbegin(v), cend(v), 42, plus<>{}, logical_not<>{});
  transform_reduce(execution::par_unseq, cbegin(v), cend(v), cbegin(v), 42, plus<>{}, minus<>{});
  uninitialized_copy(execution::par_unseq, cbegin(v), cend(v), begin(v));
  uninitialized_copy_n(execution::par_unseq, cbegin(v), size(v), begin(v));
  uninitialized_default_construct(execution::par_unseq, begin(v), end(v));
  uninitialized_default_construct_n(execution::par_unseq, begin(v), size(v));
  uninitialized_fill(execution::par_unseq, begin(v), end(v), 42);
  uninitialized_fill_n(execution::par_unseq, begin(v), size(v), 42);
  uninitialized_move(execution::par_unseq, begin(v), end(v), begin(v));
  uninitialized_move_n(execution::par_unseq, begin(v), size(v), begin(v));
  uninitialized_value_construct(execution::par_unseq, begin(v), end(v));
  uninitialized_value_construct_n(execution::par_unseq, begin(v), size(v));
  unique(execution::par_unseq, begin(v), end(v));
  unique(execution::par_unseq, begin(v), end(v), equal_to<>{});
  unique_copy(execution::par_unseq, cbegin(v), cend(v), begin(v));
  unique_copy(execution::par_unseq, cbegin(v), cend(v), begin(v), equal_to<>{});

  return EXIT_SUCCESS;
}
