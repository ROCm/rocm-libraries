/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.helpers.spec.h -- C99 port of three symbols from
 * rocke/helpers/spec.py:
 *
 *   Python                              C99 (this header)
 *   ---------------------------------   ---------------------------------------
 *   WarpTileBlockSizeMixin              rocke_warp_tile_block_size() +
 *     ._init_block_size()                rocke_warp_tile_init_block_size()
 *   choose_load_vec(...)               rocke_choose_load_vec(...)
 *   kernel_name_join(...)              rocke_kernel_name_join(...)
 *
 * These three are pure compute / string helpers: NONE of them call the IR
 * builder (rocke_b_*). They are bit-for-bit value producers whose results are
 * later baked into the IR (e.g. choose_load_vec -> const_i32), so a
 * byte-identical IR sequence follows from byte-identical return values.
 *
 * Error model mirrors the rest of the C port: an out-param + rocke_status_t
 * return code stands in for the Python `raise ValueError`.
 */
#ifndef ROCKE_HELPER_ROCKE_HELPERS_SPEC_H
#define ROCKE_HELPER_ROCKE_HELPERS_SPEC_H

#include <stddef.h>

#include "rocke/arena.h" /* rocke_arena_t (SignatureBuilder string + array storage) */
#include "rocke/ir.h" /* rocke_status_t */

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ *
 * derive_block_size / WarpTileBlockSizeMixin
 * ------------------------------------------------------------------ *
 *
 * Python:
 *
 *   def derive_block_size(tile, wave_size) -> int:
 *       return int(tile.warp_m) * int(tile.warp_n) * int(tile.warp_k)
 *              * int(wave_size)
 *
 *   class WarpTileBlockSizeMixin:
 *       def _init_block_size(self):
 *           if getattr(self, "block_size", 0) == 0:
 *               object.__setattr__(self, "block_size",
 *                   derive_block_size(self.tile, self.wave_size))
 *
 * The mixin only ever (a) derives a block size from the warp grid and the
 * wave size, and (b) installs it iff the field is still its 0 sentinel. With
 * no Python object model in C the two pieces split cleanly:
 *
 *   - rocke_warp_tile_block_size() is the pure derive_block_size().
 *   - rocke_warp_tile_init_block_size() is the lazy "derive me" guard: it
 *     returns the existing value when non-zero, else the derived value.
 */

/* derive_block_size(tile, wave_size):
 *   warp_m * warp_n * warp_k * wave_size  (signed 32-bit, like Python int()). */
int rocke_warp_tile_block_size(int warp_m, int warp_n, int warp_k, int wave_size);

/* WarpTileBlockSizeMixin._init_block_size(): if current_block_size == 0,
 * return the derived block size; otherwise return current_block_size
 * unchanged. (The frozen-dataclass object.__setattr__ write is the caller's
 * job; this returns the value to store.) */
int rocke_warp_tile_init_block_size(
    int current_block_size, int warp_m, int warp_n, int warp_k, int wave_size);

/* ------------------------------------------------------------------ *
 * IOSpecRule / validate_io
 * ------------------------------------------------------------------ *
 *
 * Python:
 *
 *   @dataclass(frozen=True)
 *   class IOSpecRule:
 *       dtype: str
 *       block_size: int
 *       vec: int
 *       n_per_block: Optional[int] = None
 *       max_elems_per_thread: Optional[int] = None
 *       allowed_dtypes: Tuple[str, ...] = ("f16", "fp16", "bf16")
 *       allowed_block_sizes: Tuple[int, ...] = (64, 128, 256, 512, 1024)
 *       allowed_vecs: Tuple[int, ...] = (2, 4, 8)
 *
 *   def validate_io(rule) -> Tuple[bool, str]:
 *       if rule.dtype not in rule.allowed_dtypes:
 *           return False, f"unsupported dtype {rule.dtype!r}"
 *       if rule.block_size not in rule.allowed_block_sizes:
 *           return False, f"block_size {rule.block_size} not in
 *                          {set(rule.allowed_block_sizes)}"
 *       if rule.vec not in rule.allowed_vecs:
 *           return False, f"vec {rule.vec} not in {set(rule.allowed_vecs)}"
 *       if rule.n_per_block is not None:
 *           chunk = rule.block_size * rule.vec
 *           if rule.n_per_block % chunk:
 *               return False, f"n_per_block ({rule.n_per_block}) must be
 *                  divisible by block_size*vec ({chunk})"
 *           if rule.max_elems_per_thread is not None:
 *               elems = rule.n_per_block // rule.block_size
 *               if elems > rule.max_elems_per_thread:
 *                   return False, f"elems_per_thread {elems} >
 *                      {rule.max_elems_per_thread}; pick a larger block_size or
 *                      a multi-pass kernel"
 *       return True, "ok"
 *
 * The frozen dataclass is a plain value struct here. The Optional[int] fields
 * (n_per_block / max_elems_per_thread) become a paired "<field>_set" flag + the
 * value, since C has no None. The allowed_* tuples are NULL-terminated /
 * counted arrays; pass NULL (with count 0) to take the Python default tuple.
 *
 * validate_io returns the bool `ok`. *out_reason (if non-NULL) receives an
 * arena-owned formatted reason string ("ok" on success). These reason strings
 * surface only through ValueError messages -- they never enter the IR -- so the
 * downstream emitted code is byte-identical regardless. The diagnostic
 * `set(...)` spelling is reproduced byte-for-byte for the two default tuples
 * (the only ones in practice); a custom allowed_* tuple falls back to its given
 * order inside the braces. */

/* One IOSpecRule. Mirrors the frozen dataclass field-for-field. */
typedef struct rocke_io_spec_rule
{
    const char* dtype;
    int block_size;
    int vec;

    /* Optional[int] n_per_block: n_per_block_set==0 means Python None. */
    int n_per_block_set;
    int n_per_block;

    /* Optional[int] max_elems_per_thread: ..._set==0 means Python None. */
    int max_elems_per_thread_set;
    int max_elems_per_thread;

    /* allowed_dtypes default = {"f16","fp16","bf16"} when allowed_dtypes==NULL. */
    const char* const* allowed_dtypes;
    size_t num_allowed_dtypes;

    /* allowed_block_sizes default = {64,128,256,512,1024} when NULL. */
    const int* allowed_block_sizes;
    size_t num_allowed_block_sizes;

    /* allowed_vecs default = {2,4,8} when NULL. */
    const int* allowed_vecs;
    size_t num_allowed_vecs;
} rocke_io_spec_rule_t;

/* Initialise `rule` with the dataclass defaults and the three required
 * positional fields (dtype / block_size / vec). The Optional fields are set to
 * "None" (the *_set flags cleared) and the allowed_* tuples are left NULL so
 * validate_io uses the Python defaults. dtype is referenced as-is (not copied);
 * keep it alive for the rule's use. */
void rocke_io_spec_rule_init(rocke_io_spec_rule_t* rule,
                             const char* dtype,
                             int block_size,
                             int vec);

/* validate_io(rule): returns 1 (ok) / 0 (reject). *out_reason (if non-NULL)
 * receives an arena-owned reason string; on the success path it is the static
 * "ok". `arena` owns any formatted reject string; pass a live arena. On a
 * formatting OOM the reason is left as a best-effort static fallback and the
 * predicate result is still returned faithfully. */
int rocke_validate_io(rocke_arena_t* arena,
                      const rocke_io_spec_rule_t* rule,
                      const char** out_reason);

/* ------------------------------------------------------------------ *
 * choose_load_vec
 * ------------------------------------------------------------------ *
 *
 * Pick the widest fp16 global-load vector width for a GEMM block tile.
 * Returns the largest v in (8, 4, 2, 1) such that v divides tile_k and both
 * (tile_m*tile_k)//v and (tile_n*tile_k)//v are >= block_size and divisible
 * by it. On no usable width the Python raises ValueError; here we return
 * ROCKE_ERR_VALUE and leave *out_vec untouched. On success *out_vec holds the
 * chosen width and ROCKE_OK is returned. */
rocke_status_t
    rocke_choose_load_vec(int tile_m, int tile_n, int tile_k, int block_size, int* out_vec);

/* ------------------------------------------------------------------ *
 * kernel_name_join
 * ------------------------------------------------------------------ *
 *
 * Python:
 *
 *   def kernel_name_join(prefix, *parts, flags=None) -> str:
 *       body = "_".join(p for p in (prefix, *parts) if p)
 *       if flags:
 *           for name, on in flags.items():
 *               if on:
 *                   body += f"_{name}"
 *       return body.replace("/", "_")
 *
 * `parts` is an ordered NULL-or-empty-skipping list. `flags` is an ordered
 * (name, on) list; entries with on != 0 contribute an "_<name>" suffix in
 * iteration order -- matching CPython 3.7+ dict insertion order, which the
 * callers rely on. Every "/" in the final string is rewritten to "_".
 *
 * The result is written NUL-terminated into `out` (capacity `out_cap`). On
 * success returns ROCKE_OK and, if out_len != NULL, sets *out_len to the byte
 * length (excluding the NUL). If the buffer is too small returns
 * ROCKE_ERR_VALUE and writes nothing usable. */
rocke_status_t rocke_kernel_name_join(const char* prefix,
                                      const char* const* parts,
                                      size_t num_parts,
                                      const char* const* flag_names,
                                      const int* flag_on,
                                      size_t num_flags,
                                      char* out,
                                      size_t out_cap,
                                      size_t* out_len);

/* ------------------------------------------------------------------ *
 * ptr_type_str / sig_param / sig_scalar
 * ------------------------------------------------------------------ *
 *
 * Python:
 *
 *   def ptr_type_str(dtype, addr_space="global") -> str:
 *       canon = "f16" if dtype in ("f16", "fp16") else dtype
 *       return f"ptr<{canon}, {addr_space}>"
 *
 *   def sig_param(name, dtype, addr_space="global") -> dict:
 *       return {"name": name, "type": ptr_type_str(dtype, addr_space)}
 *
 *   def sig_scalar(name, ty) -> dict:
 *       if ty not in ("i32", "i64", "f32"):
 *           raise ValueError(f"unsupported scalar arg type {ty!r}")
 *       return {"name": name, "type": ty}
 *
 * One manifest signature element is a {"name","type"} pair of strings. In C a
 * "dict" with exactly those two string keys is just this struct; the list of
 * dicts the Python `<op>_signature` returns is a contiguous array of these. */
typedef struct rocke_sig_entry
{
    const char* name; /* arena-owned copy of the param name */
    const char* type; /* arena-owned copy of the type string */
} rocke_sig_entry_t;

/* ptr_type_str(dtype, addr_space): writes "ptr<<canon>, <addr_space>>" into the
 * arena and returns it ("f16"/"fp16" canonicalise to "f16"; every other dtype
 * passes through verbatim). addr_space NULL is treated as "global" (the Python
 * default). Returns NULL on OOM (or on NULL dtype). */
const char* rocke_ptr_type_str(rocke_arena_t* arena, const char* dtype, const char* addr_space);

/* sig_param(name, dtype, addr_space): one pointer-kind entry. The name and the
 * ptr<...> type string are both copied into the arena. addr_space NULL =>
 * "global". On OOM the entry fields are left NULL; returns ROCKE_OK on success,
 * ROCKE_ERR_OOM on allocation failure. */
rocke_status_t rocke_sig_param(rocke_arena_t* arena,
                               const char* name,
                               const char* dtype,
                               const char* addr_space,
                               rocke_sig_entry_t* out);

/* sig_scalar(name, ty): one scalar entry. `ty` must be one of "i32"/"i64"/"f32"
 * (the Python ValueError is mapped to ROCKE_ERR_VALUE, with the entry untouched).
 * name and ty are copied into the arena. */
rocke_status_t rocke_sig_scalar(rocke_arena_t* arena,
                                const char* name,
                                const char* ty,
                                rocke_sig_entry_t* out);

/* ------------------------------------------------------------------ *
 * SignatureBuilder
 * ------------------------------------------------------------------ *
 *
 * Python:
 *
 *   @dataclass
 *   class SignatureBuilder:
 *       _items: List[dict] = field(default_factory=list)
 *       def ptr(self, name, dtype, addr_space="global") -> SignatureBuilder:
 *           self._items.append(sig_param(name, dtype, addr_space)); return self
 *       def scalar(self, name, ty) -> SignatureBuilder:
 *           self._items.append(sig_scalar(name, ty)); return self
 *       def extend(self, items) -> SignatureBuilder:
 *           self._items.extend(items); return self
 *       def build(self) -> List[dict]:
 *           return list(self._items)
 *
 * The fluent return-self chaining maps to a stateful builder threaded through
 * each call. C has no exceptions, so a failed .ptr()/.scalar()/.extend() sets a
 * sticky status on the builder; subsequent calls become no-ops (mirroring the
 * "an exception aborts the chain" Python behaviour) and .build() surfaces it.
 *
 * Storage (the growing _items list and every copied string) lives in the arena
 * passed to rocke_signature_builder_init; nothing is freed per-call. */
typedef struct rocke_signature_builder
{
    rocke_arena_t* arena; /* backing storage for entries + strings */
    rocke_sig_entry_t* items; /* arena array, grows by reallocation     */
    size_t count; /* number of populated entries            */
    size_t cap; /* allocated slots                         */
    rocke_status_t status; /* sticky: first failure aborts the chain  */
} rocke_signature_builder_t;

/* Initialise an empty builder backed by `arena`. Returns ROCKE_OK, or
 * ROCKE_ERR_VALUE if either pointer is NULL. */
rocke_status_t rocke_signature_builder_init(rocke_signature_builder_t* sb, rocke_arena_t* arena);

/* .ptr(name, dtype, addr_space): append a pointer entry. addr_space NULL =>
 * "global". Returns `sb` for chaining; on failure the sticky status is set and
 * the builder is left unchanged from this call onward. */
rocke_signature_builder_t* rocke_signature_builder_ptr(rocke_signature_builder_t* sb,
                                                       const char* name,
                                                       const char* dtype,
                                                       const char* addr_space);

/* .scalar(name, ty): append a scalar entry (ty in i32/i64/f32). */
rocke_signature_builder_t*
    rocke_signature_builder_scalar(rocke_signature_builder_t* sb, const char* name, const char* ty);

/* .extend(items): append a pre-built array of `n` entries (the {name,type}
 * strings are referenced as-is; the Python list.extend likewise stores the same
 * dict objects, not copies). */
rocke_signature_builder_t* rocke_signature_builder_extend(rocke_signature_builder_t* sb,
                                                          const rocke_sig_entry_t* items,
                                                          size_t n);

/* .build(): expose the accumulated entries. Returns ROCKE_OK and sets *out_items /
 * *out_count to the (arena-owned) array; on a prior sticky failure returns that
 * status and leaves the out-params untouched. The Python `list(self._items)`
 * makes a shallow copy; here the caller may treat the array as read-only for the
 * arena's lifetime. */
rocke_status_t rocke_signature_builder_build(const rocke_signature_builder_t* sb,
                                             const rocke_sig_entry_t** out_items,
                                             size_t* out_count);

/* ------------------------------------------------------------------ *
 * ceil_div_grid
 * ------------------------------------------------------------------ *
 *
 * Python:
 *
 *   def ceil_div_grid(*dims) -> Tuple[int, int, int]:
 *       if not (1 <= len(dims) <= 3):
 *           raise ValueError(...)
 *       out = []
 *       for total, tile in dims:
 *           if int(tile) <= 0:
 *               raise ValueError(f"tile must be positive, got {tile}")
 *           out.append((int(total) + int(tile) - 1) // int(tile))
 *       while len(out) < 3:
 *           out.append(1)
 *       return (out[0], out[1], out[2])
 *
 * `dims` is 1..3 (total, tile) pairs, passed as parallel arrays. Each axis is
 * ceil(total/tile); unspecified trailing axes default to 1. A non-positive tile
 * or a num_dims outside [1,3] is the Python ValueError -> ROCKE_ERR_VALUE (out is
 * left untouched). On success out[0..2] hold (x, y, z). */
rocke_status_t
    rocke_ceil_div_grid(const int* totals, const int* tiles, size_t num_dims, int out[3]);

/* ------------------------------------------------------------------ *
 * spec reason-string copy
 * ------------------------------------------------------------------ *
 *
 * Copy a NUL-terminated reject/validation reason `msg` into the caller's
 * `reason` buffer of capacity `cap`, truncating to fit and always
 * NUL-terminating. A NULL buffer or zero capacity is a no-op. These reason
 * strings surface only through host-side ValueError-style diagnostics; they
 * never enter the emitted IR, so they have no effect on generated code. Shared
 * so the family validators do not each carry an identical local copy. */
void rocke_spec_set_reason(char* reason, size_t cap, const char* msg);

#ifdef __cplusplus
}
#endif

#endif /* ROCKE_HELPER_ROCKE_HELPERS_SPEC_H */
