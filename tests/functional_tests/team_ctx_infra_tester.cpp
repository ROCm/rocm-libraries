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

#include "team_ctx_infra_tester.hpp"

#include <stdlib.h>

#include <rocshmem/rocshmem.hpp>

using namespace rocshmem;

/* this constant should equal ROCSHMEM_MAX_NUM_TEAMS-1 */
#define NUM_TEAMS 39

rocshmem_team_t team_world_dup[NUM_TEAMS];

/******************************************************************************
 * DEVICE TEST KERNEL
 *****************************************************************************/
__global__ void TeamCtxInfraTest(ShmemContextType ctx_type,
                                 rocshmem_team_t *team) {
  __shared__ rocshmem_ctx_t ctx1, ctx2, ctx3;
  __shared__ rocshmem_ctx_t ctx[NUM_TEAMS];

  rocshmem_wg_init();

  /**
   * Test 1: Assert team infos of different ctxs
   * from the same team are the same.
   */

  rocshmem_wg_team_create_ctx(team[0], ctx_type, &ctx1);
  assert (nullptr != ctx1.ctx_opaque);
  rocshmem_wg_team_create_ctx(team[0], ctx_type, &ctx2);
  assert (nullptr != ctx2.ctx_opaque);
  rocshmem_wg_ctx_destroy(&ctx1);
  rocshmem_wg_team_create_ctx(team[0], ctx_type, &ctx3);
  assert (nullptr != ctx3.ctx_opaque);

  __syncthreads();

  if (ctx3.team_opaque != ctx2.team_opaque) {
    printf("Incorrect for teams of ctx2 and ctx3 to not equal each other\n");
    abort();
  }

  rocshmem_wg_ctx_destroy(&ctx2);
  rocshmem_wg_ctx_destroy(&ctx3);

  __syncthreads();

  /**
   * Test 2: Assert team infos of different ctxs
   * from different teams are different.
   */
  for (int team_i = 0; team_i < NUM_TEAMS; team_i++) {
    rocshmem_wg_team_create_ctx(team[team_i], ctx_type, &ctx[team_i]);
    assert (nullptr != ctx.ctx_opaque);
  }

  if (ctx[0].team_opaque == ctx[NUM_TEAMS - 1].team_opaque) {
    printf(
        "Incorrect for ctx[0] team and ctx[NUM_TEAMS-1] to equal each other\n");
    abort();
  }

  __syncthreads();

  for (int team_i = 0; team_i < NUM_TEAMS; team_i++) {
    rocshmem_wg_ctx_destroy(&ctx[team_i]);
  }

  rocshmem_wg_finalize();
}

/******************************************************************************
 * HOST TESTER CLASS METHODS
 *****************************************************************************/
TeamCtxInfraTester::TeamCtxInfraTester(TesterArguments args) : Tester(args) {}

TeamCtxInfraTester::~TeamCtxInfraTester() {}

void TeamCtxInfraTester::resetBuffers(uint64_t size) {}

void TeamCtxInfraTester::preLaunchKernel() {
  int n_pes = rocshmem_team_n_pes(ROCSHMEM_TEAM_WORLD);

  for (int team_i = 0; team_i < NUM_TEAMS; team_i++) {
    team_world_dup[team_i] = ROCSHMEM_TEAM_INVALID;
    rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                                 &team_world_dup[team_i]);
    if (team_world_dup[team_i] == ROCSHMEM_TEAM_INVALID) {
      printf("Team %d is invalid!\n", team_i);
      abort();
    }
  }

  /* Assert the failure of a new team creation. */
  rocshmem_team_t new_team = ROCSHMEM_TEAM_INVALID;
  rocshmem_team_split_strided(ROCSHMEM_TEAM_WORLD, 0, 1, n_pes, nullptr, 0,
                               &new_team);
  if (new_team != ROCSHMEM_TEAM_INVALID) {
    printf("new team is not invalid\n");
    abort();
  }
}

void TeamCtxInfraTester::launchKernel(dim3 gridSize, dim3 blockSize, int loop,
                                      uint64_t size) {
  size_t shared_bytes = 0;

  /* Copy array of teams to device */
  rocshmem_team_t *teams_on_device;
  CHECK_HIP(hipMalloc(&teams_on_device, sizeof(rocshmem_team_t) * NUM_TEAMS));
  CHECK_HIP(hipMemcpy(teams_on_device, team_world_dup,
            sizeof(rocshmem_team_t) * NUM_TEAMS, hipMemcpyHostToDevice));

  hipLaunchKernelGGL(TeamCtxInfraTest, gridSize, blockSize, shared_bytes,
                     stream, _shmem_context, teams_on_device);

  CHECK_HIP(hipFree(teams_on_device));
}

void TeamCtxInfraTester::postLaunchKernel() {
  for (int team_i = 0; team_i < NUM_TEAMS; team_i++) {
    rocshmem_team_destroy(team_world_dup[team_i]);
  }
}

void TeamCtxInfraTester::verifyResults(uint64_t size) {}
