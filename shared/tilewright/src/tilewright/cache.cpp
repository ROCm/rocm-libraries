// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "cache.hpp"

#include "model_impl.hpp"

#include <new>

namespace tilewright {
namespace detail {

namespace {

constexpr std::size_t kDefaultCacheLimit = std::size_t{1} << 30;

}  // namespace

CacheBudget::CacheBudget() {
  const long long mib = env_knobs().cache_mib;
  limit_              = mib >= 0 ? static_cast<std::size_t>(mib) << 20 : kDefaultCacheLimit;
}

CacheBudget& CacheBudget::instance() {
  alignas(CacheBudget) static unsigned char storage[sizeof(CacheBudget)];
  static CacheBudget* const budget = new (storage) CacheBudget();
  return *budget;
}

void CacheBudget::list(CacheEntry& e, std::size_t bytes) noexcept {
  if (e.listed_) {
    used_ -= e.bytes_;
  } else {
    e.index_  = entries_.size();
    e.listed_ = true;
    entries_.push_back(&e);
  }
  e.bytes_ = bytes;
  used_ += bytes;
  e.last_use_.store(clock_.fetch_add(1, std::memory_order_relaxed) + 1, std::memory_order_relaxed);
}

void CacheBudget::unlist(CacheEntry& e) noexcept {
  CacheEntry* last   = entries_.back();
  entries_[e.index_] = last;
  last->index_       = e.index_;
  entries_.pop_back();
  used_ -= e.bytes_;
  e.bytes_  = 0;
  e.listed_ = false;
}

void CacheBudget::trim(const CacheEntry* keep) noexcept {
  while (used_ > limit_) {
    CacheEntry* victim   = nullptr;
    std::uint64_t oldest = 0;
    for (CacheEntry* e : entries_) {
      if (e == keep) continue;
      const std::uint64_t t = e->last_use_.load(std::memory_order_relaxed);
      if (victim == nullptr || t < oldest) {
        victim = e;
        oldest = t;
      }
    }
    if (victim == nullptr) return;
    unlist(*victim);
    victim->drop();
  }
}

void CacheBudget::forget(CacheEntry& e) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (e.listed_) unlist(e);
}

void CacheBudget::set_limit(std::size_t bytes) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  limit_ = bytes;
  trim(nullptr);
}

std::size_t CacheBudget::limit() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return limit_;
}

std::size_t CacheBudget::usage() const noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  return used_;
}

}  // namespace detail

void set_cache_limit(std::size_t bytes) noexcept {
  detail::CacheBudget::instance().set_limit(bytes);
}

std::size_t cache_limit() noexcept { return detail::CacheBudget::instance().limit(); }

std::size_t cache_usage() noexcept { return detail::CacheBudget::instance().usage(); }

}  // namespace tilewright
