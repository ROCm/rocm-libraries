// Copyright (c) 2016-2022, NVIDIA CORPORATION. All rights reserved.
// Modifications Copyright (c) Microsoft Corporation.
// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
// Licensed under the MIT License.

#ifndef ROCSHMEM_UTILS_HPP_
#define ROCSHMEM_UTILS_HPP_

#include <chrono>
#include <cstdint>
#include <cstdio>

#define ERROR(...) { fprintf(stderr, __VA_ARGS__); abort(); }

#ifdef ROCSHMEM_ENABLE_TRACE
#define TRACE(...) printf(__VA_ARGS__)
#else
#define TRACE(...)
#endif

#if defined ROCSHMEM_ENABLE_INFO
#define INFO(FLAGS, ...)printf(__VA_ARGS__)
#else
#define INFO(...)
#endif

namespace rocshmem {

struct Timer {
  std::chrono::steady_clock::time_point start_;
  int timeout_;

  Timer(int timeout = -1);

  ~Timer();

  /// Returns the elapsed time in microseconds.
  int64_t elapsed() const;

  void set(int timeout);

  void reset();

  void print(const std::string& name);
};

struct ScopedTimer : public Timer {
  const std::string name_;

  ScopedTimer(const std::string& name);

  ~ScopedTimer();
};

std::string getHostName(int maxlen, const char delim);

// PCI Bus ID <-> int64 conversion functions
std::string int64ToBusId(int64_t id);
int64_t busIdToInt64(const std::string busId);

uint64_t getHash(const char* string, int n);
uint64_t getHostHash();
uint64_t getPidHash();
void getRandomData(void* buffer, size_t bytes);

struct netIf {
  char prefix[64];
  int port;
};

int parseStringList(const char* string, struct netIf* ifList, int maxList);
bool matchIfList(const char* string, int port, struct netIf* ifList, int listSize, bool matchExact);

template <class T>
inline void hashCombine(std::size_t& hash, const T& v) {
  std::hash<T> hasher;
  hash ^= hasher(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
}

struct PairHash {
 public:
  template <typename T, typename U>
  std::size_t operator()(const std::pair<T, U>& x) const {
    std::size_t hash = 0;
    hashCombine(hash, x.first);
    hashCombine(hash, x.second);
    return hash;
  }
};

}  // namespace rocshmem

#endif // ROCSHMEM_UTILS_HPP
