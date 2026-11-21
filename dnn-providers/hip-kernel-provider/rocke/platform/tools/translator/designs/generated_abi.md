# Generated C ABI

Decided 2026-10-25 after translating valu, counted and aux against their
hand headers: every hole there was a C ABI shape the hand port invented
(count companions, a dict passed as a bare array, a C-only stage
dispatcher, a non-owning view built by a classmethod). Instead of matching
the hand header, the translator derives the header from the Python module.
Future ports have no hand header at all, so this is the path they need
anyway.

## What changes

Today: hand header -> `mine_maps.py` -> sigmap entry -> translator shim.
Now, for a module's **own** exports: Python -> `tr/abi.py` -> sigmap-format
entry + header text -> the same shim code. Calls into modules that are not
regenerated still use the mined sigmap.

## Export set

Everything in the module's `__all__`, plus the methods of exported classes.
Nothing outside `__all__` gets a C twin. The hand ABI left some host-only
helpers in Python (aux launch grids, `attn_bwd_aux_params`); they become
extra C functions, which no caller has to use.

## Rules

Fixed policy, one rule per Python shape. Naming keeps the hand port's style
so most declarations come out unchanged.

| Python | C |
|---|---|
| module `_attention_bwd_x` / `attention_bwd_x` | prefix `rocke_attn_bwd_` (per-package table in `tr/abi.py`) |
| function `f` | `rocke_attn_bwd_f`, a leading `attn_bwd_` in `f` dropped |
| `build_f(spec, *, arch) -> KernelDef` | `rocke_kernel_def_t* rocke_build_f_new(b, spec, arch)`: inits `b`, NULL + builder error on a raise |
| class `C` | `rocke_attn_bwd_<snake C>_t`, a leading `AttnBwd` dropped; dataclass fields in order |
| `C.__init__` / `C.__post_init__` / method `m` / classmethod `m` | `_init` / `_validate` / `_<m>` with `self` first / `_<m>` without `self` |
| `IRBuilder` | `rocke_ir_builder_t* b` |
| `Value`, `Value \| None` | `rocke_value_t*` (NULL is None) |
| `int`, `bool`, `float`, `str` | `int`, `bool`, `double`, `const char*` (NULL is the default) |
| `int \| Value` | `rocke_attn_bwd_iv_t` |
| `X \| None` field of a dataclass | `bool has_x; X x;` |
| `Sequence[T]`, `tuple[T, ...]`, `list[T]` parameter | `const T* x, int n_x` |
| `dict[str, V]` parameter | `const char* const* x_keys, const V* x_vals, int n_x` |
| `Callable[[A, B], R]` parameter | `{fn, ctx}` struct, `fn(void* ctx, A, B)` |
| dataclass parameter | `const S* x` |
| return `Value` | `rocke_value_t*`, NULL after an error |
| return `None`, function takes `b` | `bool`, false after an error |
| return `None` / `int`, no `b`, may raise | `bool` / `int` (-1), plus `char* reason, size_t reason_cap` |
| return `str` | `rocke_status_t` plus `char* out, size_t out_cap` |
| return `list[T]` | **owned**: `T** out, int* n_out`; free with `rocke_attn_bwd_free` |
| return `list[tuple[A, B]]` | owned parallel arrays `A** out_0, B** out_1, int* n_out` |
| return `dict[str, V]` | owned `const char*** out_keys, V** out_vals, int* n_out` |
| return `tuple[int, int, int]` | `int* out` (caller array of 3) |
| return a dataclass | the struct by value; tuple fields owned by it, free with `_free` |
| `tuple` of str / int constant | `extern const T NAME[NAME_N]` + `#define NAME_N` |
| `int` constant | `#define NAME value` |

Owned lists replace every count companion: the callee knows the length
after running once, so nothing runs twice (running an IR-emitting function
twice would emit its ops twice).

### Tagged values

Class families with overridden methods and declared duck-typed peers
(`data/abi/peers.json`; the dense shell's `BufferRowSlab` for a `RowSlab`)
are one tagged struct with generated dispatch: `designs/tagged_values.md`.

### Hand fields and dispatch fills (addr)

Hand callers can pass a duck-typed peer the tagged-value rule does not cover
where Python passes a class of this module (`DenseFixedAddr`'s view for a
`StridedTensorAddr`: the peer wraps the class). The Python class has no
field for it, so:

- `holes/<module>/<Class>.fields`: C-only field lines appended after the
  Python fields of the generated struct (a subclass still starts with the
  base's full layout). Zero means "not set"; generated inits memset `self`.
- `holes/<module>/<Class>_<method>.cpp` whose first lines contain
  `hole fill (dispatch`: replaces only the exported C entry of a method that
  translates. The internal `py_<Class>_<method>` stays generated; the fill
  checks the C-only field and otherwise calls it.

An ordinary fill is still used for a refused function only.

## Callers

The ABI change reaches hand-written callers: `_attention_bwd_body.cpp`,
`attention_bwd_main.cpp`, and the parity drivers of valu, counted, aux and
dense main. They are updated by hand once; the diff is the measured ABI
churn.

## Check

1. `python -m tr.abi <module> --diff` lists each declaration as same,
   changed (old and new) or new, against the hand header.
2. Gate GREEN with the generated headers and C++, the same bar as the
   translation against the hand headers.
3. Holes: target zero for valu, counted and aux.

## Result

GREEN on 2026-10-25, holes 5 to 0. Rules added while doing it: a builder
parameter when the C side needs a scratch builder (eval_layout); protocol
structs typed from the recorder's driver section; dict values that are
tuples become one value array per position; `data/abi/<module>.names.json`
keeps existing spellings.
