// Copyright (c) Microsoft Corporation.
// Modifications Copyright (c) 2025 Advanced Micro Devices, Inc, 
// Licensed under the MIT license.

#include <cstdlib>
#include <type_traits>

#include "env.hpp"
#include "utils.hpp"

template <typename T>
T readEnv(const std::string &envName, const T &defaultValue) {
  const char *envCstr = getenv(envName.c_str());
  if (envCstr == nullptr) return defaultValue;
  if constexpr (std::is_same_v<T, int>) {
    return atoi(envCstr);
  } else if constexpr (std::is_same_v<T, bool>) {
    return (std::string(envCstr) != "0");
  }
  return T(envCstr);
}

template <typename T>
void readAndSetEnv(const std::string &envName, T &env) {
  const char *envCstr = getenv(envName.c_str());
  if (envCstr == nullptr) return;
  if constexpr (std::is_same_v<T, int>) {
    env = atoi(envCstr);
  } else if constexpr (std::is_same_v<T, bool>) {
    env = (std::string(envCstr) != "0");
  } else {
    env = std::string(envCstr);
  }
}

template <typename T>
void logEnv(const std::string &envName, const T &env) {
  if (!getenv(envName.c_str())) return;
  INFO("%s=%d", envName.c_str(), env);
}

template <>
void logEnv(const std::string &envName, const std::string &env) {
  if (!getenv(envName.c_str())) return;
  INFO("%s=%s", envName.c_str(), env.c_str());
}

namespace rocshmem {

Env::Env()
    : debug(readEnv<std::string>("ROCSHMEM_DEBUG", "")),
      debugSubsys(readEnv<std::string>("ROCSHMEM_DEBUG_SUBSYS", "")),
      debugFile(readEnv<std::string>("ROCSHMEM_DEBUG_FILE", "")),
      hostid(readEnv<std::string>("ROCSHMEM_HOSTID", "")),
      socketFamily(readEnv<std::string>("ROCSHMEM_SOCKET_FAMILY", "")),
      socketIfname(readEnv<std::string>("ROCSHMEM_SOCKET_IFNAME", "")) {}

std::shared_ptr<Env> env() {
  static std::shared_ptr<Env> globalEnv = std::shared_ptr<Env>(new Env());
  static bool logged = false;
  if (!logged) {
    logged = true;
    // cannot log inside the constructor because of circular dependency
    logEnv("ROCSHMEM_DEBUG", globalEnv->debug);
    logEnv("ROCSHMEM_DEBUG_SUBSYS", globalEnv->debugSubsys);
    logEnv("ROCSHMEM_DEBUG_FILE", globalEnv->debugFile);
    logEnv("ROCSHMEM_HOSTID", globalEnv->hostid);
    logEnv("ROCSHMEM_SOCKET_FAMILY", globalEnv->socketFamily);
    logEnv("ROCSHMEM_SOCKET_IFNAME", globalEnv->socketIfname);
  }
  return globalEnv;
}

}  // namespace rocshmem
