# Tiled decode KV loading

The gfx942 and gfx950 segment kernels distinguish cache page size `P` from
compute tile size `T`. Both consume logical token tiles and stage the existing
`[T,D]` K/V LDS operands. Transfer width and split-KV segment length remain
separate target and scheduling choices. Staging is one compute tile per buffer;
MFMA and online softmax retain their existing schedules. Bounded mode adds
metadata guards and a reducer failure gate.
An internal loader-kind resolver selects strided addressing, the existing paged
descriptor, or per-token paged gather. Both engines use that decision for
validation and architecture-local K/V tile issuing. Storage conversion, transfer
width and address width remain separate choices; `uses_paged_gather` remains the
compatibility view for structural specs. No public strategy field is introduced.
For gfx942 D256/T32, V uses one tile buffer to fit the LDS budget. V has no
next-tile prefetch; the existing loop-entry wait/barrier completes prior PV
reads before the same buffer is written again. This intentionally changes
emission for a configuration that previously exceeded the target LDS limit.

- Strided KV uses independent K/V byte strides.
- Existing paged configurations retain their descriptor loader: `T == P` on
  both targets, and `T < P` on gfx942.
- The token-mapped loader supports `T=32` with `P=1,16` on both targets, and
  `P=64` on gfx950. It resolves each logical token through the shared K/V page
  table. Different lanes can reference nonadjacent physical pages.

`tile_size_override` now applies to both targets. With no override, existing
page sizes keep their previous behavior; page size 1 defaults to tile size 32.
Non-page-sized tiles append `_t<T>` to the central kernel name, including
previously unnamed gfx942 half-page variants. This fixes artifact collisions;
existing full-page names and emitted code are unchanged. The installed C spec
layout and builder signatures are unchanged.

## Admission and addressing

The initial token-mapped path admits fp16/bf16, D64/128/256, one query per
sequence, GQA ratios dividing 16, sliding windows, softcap, attention sinks,
ALiBi and query-to-query bias. These features share the existing segment
computation; only the K/V loading changes. It uses
pool-wide buffer descriptors with 64-bit base pointers and bounded i32 byte
offsets. Both contiguous `[physical_pages,P,Hkv,D]` pools must fit in
`0x7fff0000` bytes. Larger pools are rejected before cache lookup or launch;
this path does not offer a wide-address fallback or FP8 dequantization.
Sequence rounding and segment-workspace offsets are also bounded. Explicit
launches use 1..128 segments; direct default admission conservatively budgets
workspace for the policy ceiling of 128 segments.

For each cooperative transfer, the loader computes the logical token from
`tile*T + linear/D`, then derives the page-table index and within-page row
using division and remainder by `P`. Invalid speculative tokens select logical
token zero **before** looking up a page. Separately, their data offset selects
an out-of-bounds buffer offset, making the transfer supply zeros to LDS. Thus
the table guard does not depend on the data-load mask.

The preserved page-contained loaders require finite values in unused rows of
a physical page: their PV computation can multiply masked zero probabilities
by those values. The new token-mapped loader also supports NaN padding because
it zeros invalid data loads before PV. This change does not fix that inherited
page-contained limitation.

The runtime checks shapes, dtypes, contiguity, alignment, actual pool bounds,
table capacity, matching architecture, and explicit segment/reducer semantics.
It never reads device metadata to the host by default. Unmarked page-contained
legacy runtime specs retain the caller-validity contract: lengths
are in `[0,max_seqlen_k]`, query offsets are `[0,1,...,B]`, and referenced page
IDs index the physical pools. Inactive table entries need not be initialized.
Legacy C builders share this caller contract; metadata cannot be validated during
source generation. `allow_unsupported` does not bypass runtime safety checks.

The token-to-page mapping uses shared architecture-independent descriptor
helpers, mirrored by the native builders. Unmerge transforms decompose compute
coordinates and sanitized logical tokens; a descriptor with explicit byte
strides flattens the physical page, page token, KV head and dimension. The bounded
page lookup remains a checked stage between these descriptor operations: it
validates consumed physical IDs and publishes failure status before forming
their byte offsets. The descriptor `indirect` transform alone does not provide
those physical-ID checks or failure reporting.

Explicit gfx942/gfx950 paged 3D specs must expose valid page/tile geometry and
a boolean `uses_paged_gather` that agrees with it: `T > P` on gfx942, `T != P`
on gfx950. Gather also requires `use_wide_kv_load` to be present and exactly
boolean `False`, before and after retargeting. The effective tile must agree with `tile_size_override`, including
the default tile when no override is set. Address-width retargeting must preserve
the architecture, path and loader geometry. The returned gather spec is validated
again before cache lookup or launch, including segment/reducer and workspace bounds.

All explicit gfx942/gfx950 paged 3D launches also check compiled softcap,
sinks, ALiBi and QQ-bias flags against the problem and supplied feature inputs,
before and after retargeting. Flags must be booleans; enabled pointer features
require their inputs and disabled features reject supplied inputs. Sinks require
an aligned contiguous `[num_query_heads]` vector in the problem dtype; ALiBi
requires a contiguous float32 vector of the same shape. QQ bias requires rank-2
float32 storage covering the maximum local query rows, contiguous columns
and nonoverlapping rows. Partial-width bias retains zero-fill semantics when its
rows are unpadded; full-width bias may use padded rows with backing storage
covering masked tile-tail reads. All feature tensors must
share Q's HIP/CUDA device. Sink pointers are 16-byte aligned; ALiBi and QQ-bias
pointers are 4-byte aligned. The explicit segment's sliding window must equal the
problem's window on both loader kinds. Enabled
softcap accepts finite positive runtime values; disabled softcap requires zero.
These checks apply to gather and page-contained loaders, including automatic
page-size-1 launches. Matching feature-enabled specs remain valid, and
`allow_unsupported` cannot bypass agreement.
K/V storage dtype must agree with the problem. FP8 page-contained loaders retain
K/V dequantization scales; other explicit 3D loaders require numeric unit K/V
scales. Output scaling is not forwarded by the 3D adapter and must be numeric
unit scaling. Unit values exclude booleans, including on default P1 launches.
Gather-only address and workspace bounds do not restrict page-contained loaders.

The planned whole-page DMA work can add a loader kind with separate
cooperative-transfer geometry checks. Current gather restrictions remain in
place, including rejection of P16/T64. Page-ID caching must preserve safe table
reads, and page-ID substitution alone does not guarantee zero invalid K/V data.
An LDS swizzle must be shared by tile writes and QK/PV reads. Selectable loaders
at identical geometry must enter the concrete spec and compilation identity.

## Bounded metadata containment

The `attention_paged_decode_t32` candidate uses ABI `rocke-attention-paged/v2`
and enables strict boolean `metadata_containment` on both specs. Explicit marked
pairs receive the same guards in direct, prepared, internal graph and external
capture execution. Direct automatic page-size-1 launches also select a bounded
segment/reducer pair, including prepared execution and graph replay. Explicit
3D configurations also generate bounded pairs for token-gather geometry.
Runtime token-gather launches require both containment markers; disabled,
missing or mismatched markers reject before retargeting or launch, including
unsupported tuning overrides. Explicit disabled markers are not rewritten.
Unmarked page-contained legacy routes retain their caller-validity contract.
Kernel names append `_bounds_v1`; installed shared C
spec POD layouts remain unchanged. Raw native callers select the versioned
builders in [the bounded builder header](../../../platform/cpp/include/rocke/instance_attention_paged_decode_bounded.h).

Bounded mode supports gfx942/gfx950, FP16/BF16, T32, P1/16/32/64,
D64/128/256, GQA ratios 1/2/4/8/16, one query per sequence and 1..128
segments. It preserves bottom-right windows, sinks, softcap, ALiBi and QQ bias.
It rejects additive bias, top-left/unbounded bands, i64 addressing, FP8 K/V,
wide K/V and wave page-ID loading. Actual K/V tensor view spans become buffer
bounds. Host structure checks also cover rounded token/workspace/status spans
and nonoverlapping input/output/metadata views. Feature inputs must remain
separate from these views and internal workspace; raw C callers must supply
truthful allocation bounds, matching launch grids and valid lifetimes.

Each segment derives Q and scratch rows from its trusted CTA sequence index.
Live query offsets only check consistency with `[0,1,...,B]`; lengths are checked
before token arithmetic. Each used logical page index is guarded before its
load, and that same physical ID is checked and sanitized before multiplication.
Unused table tails and pages wholly excluded by the window are not audited.
Invalid lengths, offsets or consumed IDs produce NaN across every head and
dimension of the affected sequence. Unrelated sequences remain valid, and valid
zero length retains zero output, including with sinks. NaN is not a unique
error diagnostic for inputs that already contain nonfinite numeric values.

Each CTA overwrites one `[B,Hkv,S]` int32 status slot every execution, including
neutral exits. Sixty-four lane-private sticky LDS flags add 256 bytes per CTA;
a final wave reduction publishes failure after outstanding DMA completes. The
reducer scans all status slots for its sequence before reading partial output.
There is no status-clear kernel, atomic, or hidden host synchronization. Ordered
metadata changes between launches/replays are supported. Concurrent GPU writes
during reads remain the caller's synchronization responsibility.

Bounded page-contained/subpage paths use checked per-copy offsets instead of
hoisted page IDs; loader classification and DMA schedules remain unchanged.
This is a containment tradeoff with extra table/guard work and reducer status
reads. No performance improvement is claimed. `attention_3d_workspace_nbytes`
accepts the selected `tuning_spec` to account for its segment count and status.

The [containment host suite](../../tests/test_paged_metadata_containment.py)
checks range proofs, pair admission and launch signatures. The
[containment numeric suite](../../tests/test_paged_metadata_containment_numeric.py)
checks invalid-to-valid recovery on unchanged buffers across execution modes,
affected/unaffected sequences, zero lengths and Q/O canaries. The paged parity
family appends 32 bounded segment/reducer cases to the original 66 configs.
Its Python goldens and native byte-parity tests run in the installed
`rocke_library_attention_cpu_pytest` gate, with the native extension required
when pybind is built. This dedicated gate is separate from the platform's
all-family byte-identity dashboard.

## Selection and validation

Use dispatch algorithm `paged_decode_t32` to request T32 explicitly. The
candidate is automatic for page size 1 and opt-in for existing page sizes, so
it preserves their default routing. Explicit `ExplicitAttention3DConfig`
accepts `tile_policy="32"` from `TILE_POLICIES_3D`. Tile tokens share whitespace
and case normalization; `resolve_tile_policy(..., path="3d")` selects the 3D
vocabulary, while the default retains the existing 2D policies.
Direct automatic execution selects its 3D specs in
`run_unified_attention_torch`, independently of the Python dispatcher. Native
3D builders consume supplied spec fields; a native problem-to-spec heuristic
API remains deferred as described in [the dispatch rules](../../dispatch/AGENTS.md).
The candidate binds without reading device metadata by default and keeps the
original device tensors live across launches. Pass
`unsafe_skip_paged_value_validation=False` to validate contents at bind time;
the option requires a boolean. That bind-time check is a snapshot; bounded device checks still run after
ordered mutation and replay. The candidate accepts only numeric unit values for `k_scale`,
`v_scale` and `out_scale` in tensor maps, binding arguments and launch overrides;
`softmax_scale` remains independently configurable. Requests carry `softcap`,
`use_sinks`, `use_alibi` and `use_qq_bias`; the problem and compiled spec must
agree. Feature tensors may be supplied in the tensor map or at binding time,
and replaced at launch while preserving enablement. Pass `qq_bias_stride_0`
in elements with query-to-query bias. The final stride must be positive, fit i32,
and match the tensor's row stride; addressed query indices must fit i32 too.
Binding may defer an omitted/zero stride until a launch override supplies it,
but a launch without that override rejects before cache/build work.
Structural validation always applies. Candidate launches use direct-call ownership with
independent eager scratch on each invocation. For reusable scratch with a fixed
eager stream and explicit close, use `prepare_unified_attention_torch` with the
ownership and graph lifecycle described in
[the strided decode documentation](../../../docs/strided-kv-decode.md).
Direct `backend="3d"` also supports page size 1.
Contiguous non-paged inputs continue to use `kv_layout="strided"`.

The paired `attention_paged_kv_emit` emitters cover both targets, both working
dtypes, D64/128/256 and P1/16/32/64. Host tests cover native byte identity at
LLVM20/22/23, golden hashes, address mapping, artifact identity and admission.
`test_paged_kv_decode_numeric.py` exercises shuffled nonadjacent pages, NaN
padding, page/tile boundaries, empty rows, output canaries, changing lengths,
streams and graph reuse against a CPU FP32 reference after input quantization.
The strided and paged regression suite remains a separate GPU gate.
No performance improvement is claimed.

Raw attention request/problem geometry, explicit configuration integers, and
segment/reducer geometry require integers excluding booleans. Integral floats,
numeric strings and conversion-only objects are rejected before selection,
kernel naming, binding or launch. Behavioral flags require booleans. Optional
wave/tile values retain their declared `None` defaults. Structural runtime specs
receive the same type checks before and after paged address-width retargeting;
matching numeric values alone do not establish agreement.

Tuning pins retain the documented lossless `KnobSpace` axis conversions and
produce typed concrete specs. Direct explicit configs require their declared
types. String tile policies retain whitespace/case normalization. The native
Python 3D dictionary adapter validates supplied types before C++ casts; its
partial-dictionary API preserves missing/`None` defaults. The typed C ABI layout
and valid kernel emission are unchanged.

### Optional LSE with bounded decode

Admitted gfx942/gfx950 Python decode paths accept `emit_lse=True` and a caller-owned
contiguous FP32 `lse` tensor shaped `[total_q, num_query_heads, 1]`, on the same
device as Q/O and disjoint from all input/output views. Both default P1 selection
and explicit T32 paged pairs propagate the requested output. Request, problem and
reducer must agree; missing or stray buffers and malformed boolean flags reject
before compilation. LSE remains unsupported with sinks, strided KV, and paths
that cannot write it.

The reducer arguments are `output_ptr`, `segm_output_ptr`, `segm_max_ptr`,
`segm_expsum_ptr`, `seq_lens_ptr`, followed by `metadata_status_ptr`, `num_seqs`
when containment is enabled, and finally `lse_ptr` when LSE is enabled. Kernel
names and cache identities distinguish both flags. Invalid consumed metadata
writes NaN to both O and LSE for the affected sequence; valid keyless rows write
O=0 and natural-log LSE=-inf. Window-excluded pages remain unaudited. An ordered
valid replay recovers both outputs, and graph ownership retains the LSE buffer.

Native C++ 3D builders accept absent/false LSE only and explicitly reject enabled
or malformed LSE fields. The dedicated 98-case native paged registry remains
LSE-off. Strict C++ lowering of Python-authored bounded-LSE IR and representative
bounded-LSE goldens are separate coverage from native kernel construction.
