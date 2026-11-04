# Layer files and layer checks

The folder `test-plans/sdpa-fwd/layers` holds three research files. They
describe the forward SDPA settings in three layers:

| File | Layer | What it holds |
|---|---|---|
| `L0-axes.yaml` | 0 | Every setting and value that has a meaning for attention. |
| `L1-definition.yaml` | 1 | Rules that remove combinations that have no meaning. |
| `L2-frontend.yaml` | 2 | Rules that remove combinations that the hipDNN frontend cannot state. |

The generator does not read these files. Two scripts in this folder read them.

| Script | What it does |
|---|---|
| `layer_counts.py` | Counts the combinations and the valid value pairs of each layer. |
| `chain_check.py` | Checks that Layer 0, then the Layer 1 and Layer 2 rules, give the values of `model.yaml`. |

`chain-map.yaml` tells `chain_check.py` how a model value maps to a Layer 0
value when the two ids are not the same.

## Run the checks

Run the commands from the folder `dnn-providers/integration-tests/scripts`.
You need Python 3 and the PyYAML package.

Count the combinations and pairs of each layer. The script writes a JSON
summary to standard output. It takes about one minute.

```bash
python3 sdpa_case_flow_poc/layers/layer_counts.py ../test-plans/sdpa-fwd/layers
```

To also write every valid pair of each layer to a CSV file, add
`--pairs-out <file>`. The CSV has the columns
`layer,axis_a,value_a,axis_b,value_b`.

Check the chain from Layer 0 to the model. The exit code is 0 when the output
has no `ERROR` line.

```bash
python3 sdpa_case_flow_poc/layers/chain_check.py ../test-plans/sdpa-fwd \
    sdpa_case_flow_poc/layers/chain-map.yaml
```

If you change `model.yaml`, `rules.yaml` or a Layer file, run the chain check
in the same change.
