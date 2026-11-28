<!--
Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
SPDX-License-Identifier: MIT
-->

# Decision: rocKE's own worker plugin, not pytest-xdist

**Decision.** A CI pytest entry runs its tests in worker processes through
`rocke.tests.ctest_workers` (`--rocke-workers N`), not through pytest-xdist.
[TESTING.md §5.0.2](../../TESTING.md#502-ci-shards-and-rocke-item-sharding)
says how the plugin works.

**Why.** The first three rows below decide it: pytest-xdist, as it ships, fails
them. It needs our own code for two more, and handles a stalled test better.

| Requirement | `ctest_workers` | pytest-xdist |
|---|---|---|
| No extra package in the CI test environment | pytest only | **fails**: TheRock's test environment (its common `requirements-test.txt` and the provider's `requirements-test-hipkernelprovider.txt`) has no pytest-xdist or execnet |
| A `rocke_run_alone` test runs while no other process of its entry exists, so none holds device memory | a last process, started after every worker has exited | **fails**: workers live until the session ends, so a grouped or last-scheduled test still shares the device with idle workers; it needs a second pytest command or entry |
| The CI shard plugin prints its split line and cost-mark check | each worker prints them; the run relays its lines | **fails**: workers print no terminal summary, and the controller has no split, so both are lost |
| A per-test time limit (pytest-timeout, thread method) ends a stalled test | pytest-timeout's stack dump names the test; the run names the worker and fails, and **that worker's remaining tests do not run** | better: the controller names the test, replaces the worker, and the other tests run |
| Each worker gets its share of the CPUs (affinity on Linux, thread-count variables), set before conftests load | yes | needs a plugin of our own |
| Run records count each test's collection once | worker 0 records it | needs a port to `PYTEST_XDIST_WORKER` |
| Windows | yes | yes |

Neither runs a worker on a GPU of its own: CI gives a shard one GPU, which the
workers share.

**Revisit when** rocKE installs its own test requirements file in the provider's
test artifact (TheRock's provider job has a TODO to read it from there), so
adding pytest-xdist is rocKE's own change. A move then also needs the run-alone
tests in a command of their own, a small plugin for CPU shares and run records,
and the shard summary relayed from the workers; it changes how entries run, so
it needs CI-container runs before it lands.
