# Workspace management

rocKE separates workspace description, allocation ownership, and asynchronous
lifetime:

- An operation returns `WorkspaceSpec` requirements and, when needed, a
  plan that maps the resulting tensors to kernel arguments.
- The framework constructs `WorkspaceLeasePool`, so the framework owns the
  pool object, allocator callback, and actual tensor storage.
- rocKE owns the bounded lease state machine and HIP event that prevents reuse
  before the final workspace consumer completes.

This split lets a serving framework reuse memory without making rocKE depend on
that framework's scheduler or global allocator.

## Two pool models

`WorkspacePool` remains the named-slot model for a fixed, long-lived pipeline.
A slot grows to its largest requested shape and is not exclusive. It is useful
when one cached launcher has a bounded shape and cannot execute concurrently.

`WorkspaceLeasePool` is the serving model. Its entries are anonymous
capacities rather than permanent operation names. Every acquisition is
exclusive and moves through:

```text
free -> acquired -> pending(completion event) -> free
```

An acquired entry is never returned by another `acquire`. Pending events are
normally queried without blocking at the start of acquisition and by
`reap_completed()`.

There is one safe fast path: a lease pending on a Runtime-owned event may be
reacquired by the same stream without querying the event. Same-stream FIFO
orders the new prep writes after the previous scan reads. Other streams must
observe event completion first.

## Bounds and backpressure

The lease pool requires explicit limits:

- `max_bytes` bounds all free, acquired, and pending physical storage.
- `max_cached_bytes` bounds completed storage retained for reuse.
- `max_entries` bounds fragmentation and event count.

Acquisition first reaps completed events, then chooses the smallest compatible
free entry. Free entries are evicted least-recently-used when a request needs
capacity or the cache exceeds its bound. If active/pending leases consume the
bound, `WorkspacePoolExhausted` is raised; the framework may fall back, defer
the request, or apply its own backpressure. rocKE does not silently synchronize
unrelated work.

Large one-off shapes therefore cannot make a named slot grow forever. A request
larger than `max_bytes` is rejected before allocation.

## Event ownership

After enqueueing the final workspace consumer, the caller invokes:

```python
summary = launch_final_stage(record_event=True)
lease.release_after_event(summary.completion_event)
```

The preferred path borrows the final `LaunchSummary` event already owned by
`Runtime._pending_args`; the pool queries it but does not destroy it.
`release_after(stream)` remains available when no launch event exists and
records a pool-owned event. Recording after the final consumer is the ordering
point; same-stream FIFO guarantees all earlier prep/scan accesses are covered.
No host synchronization or duplicate KDA event is inserted.

If launch submission fails after earlier stages were enqueued, the caller must
first synchronize that stream and then call `lease.release_completed()`.
Releasing without either a completion event or an external synchronization is
invalid.

The event therefore has two coordinated responsibilities: Runtime owns and
destroys it after protecting packed launch arguments, while the lease pool uses
the same completion state to decide when storage can be reassigned.

## Framework-owned allocation

`WorkspaceLeasePool` accepts an `allocator(WorkspaceSpec)` callback. For
example, vLLM supplies a callback that calls `torch.empty`; the
`RockeKdaEngine` instance holds the pool and therefore owns all allocation
references. rocKE never stores workspace in a process-global shape cache.

Destroying the framework owner drops free tensors normally. `clear()` drops
free entries explicitly and refuses while leases remain acquired or pending.

## KDA workspace plan

`kda_workspace_plan(num_tiles, prep_spec, device=...)` describes two physical
allocations:

```text
kda_tiles: BF16[num_tiles * (2*C*C + 3*C*DK)]
kda_decay: FP32[num_tiles, DK]
```

The plan binds the BF16 allocation into `A`, `GK`, `GQ`, `Aqk`, and `Kt`
views and exposes the FP32 decay view. For `C=32, DK=128`, this is 29,184 bytes
per materialized chunk.

The vLLM KDA sequence is:

```python
plan = kda_workspace_plan(num_tiles, prep.spec, device=q.device)
lease = engine_workspace_pool.acquire(plan.specs)
workspace = plan.bind(lease.tensors)

launch_prep(workspace, stream=stream)
summary = launch_scan(workspace, stream=stream, record_event=True)
lease.release_after_event(summary.completion_event)
```

The plugin defaults to a bounded per-engine pool and exposes environment
overrides for total bytes, cached bytes, and entry count. Pool exhaustion is an
unsupported rocKE request, allowing the existing Triton fallback rather than
unbounded allocation.
