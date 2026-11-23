// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

namespace tilewright {
namespace detail {

// Data the engine builds on demand and can build again (a cell's fp32
// weights, a candidate set's cache of one cell), counted against the
// process-wide cache limit. The owner installs new data only through
// CacheBudget::publish and must call CacheBudget::forget before the entry's
// members are destroyed.
class CacheEntry {
 public:
  CacheEntry()                             = default;
  CacheEntry(const CacheEntry&)            = delete;
  CacheEntry& operator=(const CacheEntry&) = delete;

  // Releases the data. Runs with the budget's lock held, so it must not call
  // into the budget.
  virtual void drop() noexcept = 0;

 protected:
  ~CacheEntry() = default;

 private:
  friend class CacheBudget;
  std::size_t bytes_ = 0;
  std::size_t index_ = 0;  // position in CacheBudget::entries_ while listed
  bool listed_       = false;
  std::atomic<std::uint64_t> last_use_{0};
};

// The entries holding data, and their total size. When a publish takes the
// total over the limit, the least recently used other entries are dropped
// until it fits (or only the published entry is left). Rankings that still
// hold dropped data keep it alive until they return.
class CacheBudget {
 public:
  // Never destroyed, so that models and candidate sets in static storage can
  // forget their entries at exit.
  static CacheBudget& instance();

  // Runs `install`, which stores `e`'s new data and must not throw, counts `e`
  // as `bytes` large and drops other entries while the total exceeds the
  // limit.
  template <typename Install>
  void publish(CacheEntry& e, std::size_t bytes, Install&& install) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!e.listed_ && entries_.size() == entries_.capacity())
      entries_.reserve(entries_.empty() ? 64 : 2 * entries_.size());
    install();
    list(e, bytes);
    trim(&e);
  }

  // Stops counting `e` without dropping its data.
  void forget(CacheEntry& e) noexcept;

  // Records a use of `e`'s data. Entries used since the last publish count as
  // equally recent.
  void touch(CacheEntry& e) noexcept {
    const std::uint64_t now = clock_.load(std::memory_order_relaxed);
    if (e.last_use_.load(std::memory_order_relaxed) != now)
      e.last_use_.store(now, std::memory_order_relaxed);
  }

  void set_limit(std::size_t bytes) noexcept;
  std::size_t limit() const noexcept;
  std::size_t usage() const noexcept;

 private:
  CacheBudget();
  void list(CacheEntry& e, std::size_t bytes) noexcept;
  void unlist(CacheEntry& e) noexcept;
  void trim(const CacheEntry* keep) noexcept;

  mutable std::mutex mutex_;
  std::vector<CacheEntry*> entries_;
  std::size_t used_  = 0;
  std::size_t limit_ = 0;
  std::atomic<std::uint64_t> clock_{1};
};

}  // namespace detail
}  // namespace tilewright
