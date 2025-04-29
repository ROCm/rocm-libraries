/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "util.hpp"

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "rocshmem_config.h"  // NOLINT(build/include_subdir)

namespace rocshmem {

__constant__ int* print_lock;

rocshmem_env_config rocshmem_env_;

typedef struct device_agent {
  hsa_agent_t agent;
  hsa_amd_memory_pool_t pool;
} device_agent_t;

std::vector<device_agent_t> gpu_agents;
std::vector<device_agent_t> cpu_agents;

hsa_status_t rocm_hsa_amd_memory_pool_callback(
    hsa_amd_memory_pool_t memory_pool, void* data) {
  hsa_amd_memory_pool_global_flag_t pool_flag{};

  hsa_status_t status{hsa_amd_memory_pool_get_info(
      memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &pool_flag)};

  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to get pool info: 0x%x", status);
    return status;
  }

  if (pool_flag == (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
                    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
    *static_cast<hsa_amd_memory_pool_t*>(data) = memory_pool;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t rocm_hsa_agent_callback(hsa_agent_t agent,
                                     [[maybe_unused]] void* data) {
  hsa_device_type_t device_type{};

  hsa_status_t status{
      hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type)};

  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to get device type: 0x%x", status);
    return status;
  }

  if (device_type == HSA_DEVICE_TYPE_GPU) {
    gpu_agents.emplace_back();
    gpu_agents.back().agent = agent;
    status = hsa_amd_agent_iterate_memory_pools(
        agent, rocm_hsa_amd_memory_pool_callback, &(gpu_agents.back().pool));
  }

  if (device_type == HSA_DEVICE_TYPE_CPU) {
    cpu_agents.emplace_back();
    cpu_agents.back().agent = agent;
    status = hsa_amd_agent_iterate_memory_pools(
        agent, rocm_hsa_amd_memory_pool_callback, &(cpu_agents.back().pool));
  }

  return status;
}

int rocm_init() {
  hsa_status_t status{hsa_init()};

  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to open HSA connection: 0x%x", status);
    return 1;
  }

  status = hsa_iterate_agents(rocm_hsa_agent_callback, nullptr);

  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    printf("Failure to iterate HSA agents: 0x%x", status);
    return 1;
  }

  return 0;
}

void rocm_memory_lock_to_fine_grain(void* ptr, size_t size, void** gpu_ptr,
                                    int gpu_id) {
  hsa_status_t status{
      hsa_amd_memory_lock_to_pool(ptr, size, &(gpu_agents[gpu_id].agent), 1,
                                  cpu_agents[0].pool, 0, gpu_ptr)};

  if (status != HSA_STATUS_SUCCESS) {
    printf("Failed to lock memory pool (%p): 0x%x\n", ptr, status);
    exit(-1);
  }
}


rocshmem_env_config::rocshmem_env_config() {
  char* env_value = NULL;

  env_value = getenv("ROCSHMEM_RO_DISABLE_IPC");
  if (NULL != env_value) {
    ro_disable_ipc = atoi(env_value);
  }

  env_value = getenv("ROCSHMEM_RO_PROGRESS_DELAY");
  if (nullptr != env_value) {
    ro_progress_delay = atoi(env_value);
  }

  env_value = getenv("ROCSHMEM_BOOTSTRAP_TIMEOUT");
  if (nullptr != env_value) {
    bootstrap_timeout = atoi(env_value);
  }

  env_value = getenv("ROCSHMEM_BOOTSTRAP_HOSTID");
  if (nullptr != env_value) {
    bootstrap_hostid = std::string(env_value);
  }

  env_value = getenv("ROCSHMEM_BOOTSTRAP_SOCKET_FAMILY");
  if (nullptr != env_value) {
    bootstrap_socket_family = std::string(env_value);
  }

  env_value = getenv("ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME");
  if (nullptr != env_value) {
    bootstrap_socket_ifname = std::string(env_value);
  }
}

int rocshmem_env_config::get_ro_disable_ipc() {
  return ro_disable_ipc;
}

int rocshmem_env_config::get_ro_progress_delay() {
  return ro_progress_delay;
}

int rocshmem_env_config::get_bootstrap_timeout() {
  return bootstrap_timeout;
}

std::string rocshmem_env_config::get_bootstrap_hostid() {
  return bootstrap_hostid;
}

std::string rocshmem_env_config::get_bootstrap_socket_family() {
  return bootstrap_socket_family;
}

std::string rocshmem_env_config::get_bootstrap_socket_ifname() {
  return bootstrap_socket_ifname;
}

}  // namespace rocshmem
