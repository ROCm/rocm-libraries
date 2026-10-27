# Self-test fixture root

| Path | What it is |
|---|---|
| `library/kernels/common/` | Pinned copies of the attention backward modules the test translates (`_attention_bwd_valu`, `_attention_bwd_counted`), the module whose header they use (`_attention_bwd_addr`) and the modules the translator reads beside them (`_attention_bwd_frag`, imported by valu; `_attention_bwd_paged`, which imports private addr helpers; `_attention_dense_bwd_shell`, which holds a peer class). `kernels/__init__.py` is a stub. |
| `data/facts/` | Recorded types of addr, valu and counted (`tr.record`), keyed by qualname. |
| `data/abi/` | `peers.json` and counted's kept names. |
| `data/maps/` | `sigmap.json` and `twin_map.json` mined from a tree with the hand C++ twins of these modules, cut down to the entries of addr, valu and counted, frag's `eval_layout` (the one name valu imports from it) and the IR builder methods the fixture modules call. |
| `golden/` | The expected headers and sources (`.golden`, so no formatter touches them). |

The pinned modules do not follow the live `library/`: the test checks the
translator, not the current Python. To move the fixtures to newer Python,
copy the modules, re-record their facts, and update the goldens with
`--update-goldens`.
