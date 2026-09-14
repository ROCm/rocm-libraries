# `hiprtc_file` config fixtures

One valid drop-in config and one mutation of it per rejection, consumed by
`tests/test_hiprtc_dropin.py`. `sources/` is the bundle every fixture names; its
contents are never compiled, only staged.

`valid.yaml` is the only one that loads and generates. Every other file differs
from it in exactly one place, so the test asserting the rejection is asserting
about that place and nothing else.

| Fixture | Rejected by | Because |
|---|---|---|
| `valid.yaml` | — | loads, generates, stages the bundle |
| `undeclared_field.yaml` | load | `$kernel.nosuchfield` names no KMD field; at runtime this drops the whole pack with one `LOG_ERROR`, on a machine with no config to read |
| `float_field.yaml` | load | a `float` has no single spelling — `1` and `1.0` are different types in device code — so it cannot be bound into a `-D` flag at all |
| `expression_value.yaml` | load | the substituter evaluates nothing; rendering `$kernel.block_size * 2` literally would put `64 * 2` into a flag, the half-working outcome the refusal exists to prevent |
| `non_kernel_token.yaml` | load | `$graph.` is not a binding source; nothing but `$kernel.` is |
| `non_string_define.yaml` | load | `defines` is a flat string→string map, and an unquoted YAML scalar arrives as an `int` where the flag's text belongs |
| `escaping_bundle.yaml` | load | the loader resolves `bundle` against the descriptor and refuses any source outside the descriptor tree root |
| `missing_source_file.yaml` | **generation** | loads clean, then fails at staging: the bundle does not hold the file the descriptor names |
