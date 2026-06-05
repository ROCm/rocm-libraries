# MIOpenDriver

The `MIOpenDriver` enables the user to test the functionality of any particular
layer in MIOpen in both the forward and backward direction. MIOpen is shipped with `MIOpenDriver` and its install directory is `miopen/bin` located in the install directory path.


## Building the Driver

MIOpenDriver can be build by typing:

```make MIOpenDriver``` from the ```build``` directory.


## Base Arguments
All the supported layers in MIOpen can be found by the supported `base_args` here:

``` ./bin/MIOpenDriver --help ```

The supported base arguments:

 * `conv` - Convolutions
 * `CBAInfer` - Convolution+Bias+Activation fusions for inference
 * `CAInfer` - Convolution+Activation fusions for inference
 * `pool` - Pooling
 * `lrn` - Local Response Normalization
 * `activ` - Activations
 * `softmax` - Softmax
 * `bnorm` - Batch Normalization
 * `rnn` - Recurrent Neural Networks (including LSTM and GRU)
 * `gemm` - General Matrix Multiplication
 * `ctc` - CTC Loss Function
 * `dropout` - Dropout
 * `tensorop` - Ternary Tensor Operation
 * `reduce` - Reduce
 * `layernorm` - Layer Normalization
 * `groupnorm` - Group Normalization
 * `cat` - Cat Forward Operation
 * `addlayernorm` - Add and Layer Normalization
 * `t5layernorm` - T5 Layer Normalization
 * `adam` - Adam Optimizer
 * `ampadam` - AMP Adam Optimizer
 * `adamw` - AdamW Optimizer
 * `ampadamw` - AMP AdamW Optimizer
 * `transformersadamw` - Hugging Face Transformer AdamW Optimizer
 * `transformersampaadamw` - Hugging Face Transformer AMP AdamW Optimizer
 * `getitem` - Getitem Operation
 * `reducecalculation` - Reduce Calculation
 * `rope` - Rotary Position Embedding
 * `prelu` - Parametric ReLU
 * `kthvalue` - Kthvalue Operation
 * `glu` - Gated Linear Unit
 * `softmarginloss` - Softmarginloss
 * `multimarginloss` - Multimarginloss

 These base arguments support fp32 float type, but some of the drivers suport further datatypes -- specifically, half precision (fp16), brain float16 (bfp16), and 8-bit integers (int8).
 To toggle half precision simpily add the suffix `fp16` to end of the base argument; e.g., `convfp16`.
 Likewise, to toggle brain float16 just add the suffix `bfp16`, and to use 8-bit integers add `int8`.

 Notes for this release:
  * Only convolutions support int8
  * Only reduce supports double-precision fp64
  * RNN's support fp16 but only on the HIP backend
  * CTC loss function only supports fp32

Summary of base_args meant for different datatypes and different operations:

| base_args            | Single-Precision (fp32) | Half-Precision (fp16) | Bfloat16 (bfp16)   |
| :------------------- | :---------------------: | :-------------------: | :----------------: |
| conv                 | ✓ | ✓ | ✓ |
| CBAInfer             | x | x | ✓ |
| CAInfer              | x | x | ✓ |
| pool                 | ✓ | ✓ | x |
| lrn                  | ✓ | ✓ | x |
| activ                | ✓ | ✓ | x |
| softmax              | ✓ | ✓ | x |
| bnorm                | ✓ | ✓ | ✓ |
| rnn                  | ✓ | ✓ | x |
| gemm                 | ✓ | ✓ | x |
| ctc                  | ✓ | x | x |
| dropout              | ✓ | ✓ | x |
| tensorop             | ✓ | x | x |
| reduce               | ✓ | ✓ | x |
| layernorm            | ✓ | ✓ | ✓ |
| groupnorm            | ✓ | ✓ | ✓ |
| cat                  | ✓ | ✓ | ✓ |
| addlayernorm         | ✓ | ✓ | ✓ |
| t5layernorm          | ✓ | ✓ | ✓ |
| adam                 | ✓ | ✓ | x |
| ampadam              | ✓ | x | x |
| reduceextreme        | ✓ | ✓ | ✓ |
| adamw                | ✓ | ✓ | x |
| ampadamw             | ✓ | x | x |
| transformersadamw    | ✓ | ✓ | x |
| transformersampadamw | ✓ | x | x  |
| getitem              | ✓ | ✓ | ✓ |
| reducecalculation    | ✓ | ✓ | ✓ |
| rope                 | ✓ | ✓ | ✓ |
| prelu                | ✓ | ✓ | ✓ |
| kthvalue             | ✓ | ✓ | ✓ |
| glu                  | ✓ | ✓ | ✓ |
| softmarginloss       | ✓ | ✓ | ✓ |
| multimarginloss      | ✓ | ✓ | ✓ |

## Executing MIOpenDriver

To execute from the build directory:

```./bin/MIOpenDriver *base_arg* *layer_specific_args*```

Or to execute the default configuration simpily run:

```./bin/MIOpenDriver *base_arg*```

MIOpenDriver example usages:

- Convolution with search on:

```./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2```

- Forward convolution with search off:

```./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1```

- Convolution with half or bfloat16 input type

```./bin/MIOpenDriver convfp16 -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1```
```./bin/MIOpenDriver convbfp16 -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2 -s 0 -F 1```

- Pooling with default parameters:

```./bin/MIOpenDriver pool```

- LRN with default parameters and timing on:

```./bin/MIOpenDriver lrn -t 1```

- Batch normalization with spatial fwd train, saving mean and variance tensors:

```./bin/MIOpenDriver bnorm -F 1 -n 32 -c 512 -H 16 -W 16 -m 1 -s 1```

- RNN with forward and backwards pass, no bias, bi-directional and LSTM mode

```./bin/MIOpenDriver rnn -n 4,4,4,3,3,3,2,2,2,1 -k 10 -H 512 -W 1024 -l 3 -F 0 -b 0 -r 1 -m lstm```

- Printout layer specific input arguments:

`./bin/MIOpenDriver *base_arg* -?` **OR**  `./bin/MIOpenDriver *base_arg* -h (--help)`

Note: By default the CPU verification is turned on. Verification can be disabled using `-V 0`.

## Environment Variables

### Kernel Name and Execution Time Logging

The `MIOPEN_PERFORMANCE_LOGS` environment variable enables lightweight logging of kernel names and their execution times during MIOpenDriver runs. This is useful for debugging, performance analysis, and understanding which kernels are being executed under different configurations (e.g., different `MIOPEN_FIND_MODE` or `MIOPEN_FORCE` settings). It is meant for usage only with MIOpen driver commands and is untested for direct library usage.

**Logging Levels:**

The variable supports five levels with varying detail and scope:

- **Level 0** (default): No kernel logging
- **Level 1**: Log only executed solution with **total solution time** - excludes find/search
- **Level 2**: Log **all kernels** for executed solutions individually - includes transpose/transform kernels, excludes find/search
- **Level 3**: Log only **performance configs** per solution with **total solution time** - includes find/search
- **Level 4**: Log **all kernels** individually - includes transpose/transform kernels and find/search kernels

**Usage Examples:**

```bash
# Level 4: Log all available kernel/performance config/solution information
export MIOPEN_PERFORMANCE_LOGS=4
./bin/MIOpenDriver conv -W 32 -H 32 -c 3 -k 32 -x 5 -y 5 -p 2 -q 2
```

**JSON Output Format:**

Each solution outputs a JSON object with performance configs. Each config contains aggregated timing statistics and optionally individual kernel execution data:

```json
{
  "solution": "ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC",
  "solver_id": 1234567890,
  "phase": "execution",
  "performance_configs": [
    {
      "config_name": "igemm_fwd_gtcx3_nhwc_bf16_bx0_ex1_bt128x128x32...",
      "config_descriptor": "gemm_m_per_block=128 gemm_n_per_block=128...",
      "exec_number": 1,
      "time_executions_ms": [1.52, 1.49, 1.41, 1.38],
      "time_ms": 1.42,
      "time_std_ms": 0.056,
      "time_min_ms": 1.38,
      "time_max_ms": 1.52,
      "number_of_transformations": 4,
      "kernels": [
        {
          "kernel_name": "SubTensorOpWithScalar1d",
          "time_executions_ms": [0.083, 0.104, 0.089],
          "is_transformation": true
        },
        {
          "kernel_name": "igemm_fwd_gtcx3_nhwc_bf16_bx0_ex1_bt128x128x32...",
          "time_executions_ms": [0.893, 0.763, 0.724],
          "is_transformation": false
        }
      ]
    }
  ]
}
```

**JSON Fields:**

**Solution Level:**
- `solution`: Name of the solver/algorithm
- `solver_id`: Numeric identifier for the solver
- `phase`: Either "execution" (actual computation) or "tuning" (find/search phase), "validation", "solver_tuning", or "unknown"
- `performance_configs`: Array of performance configuration results

**Performance Config Level:**
- `config_name`: Name of the main kernel or solution
- `config_descriptor`: Performance config parameters (optional)
- `exec_number`: Execution counter (starts at 1)
- `time_executions_ms`: Array of all execution times for this config
- `time_ms`: Mean execution time (computed using outlier removal)
- `time_std_ms`: Standard deviation of execution times
- `time_min_ms`: Minimum execution time
- `time_max_ms`: Maximum execution time
- `number_of_transformations`: Count of transformation/transpose kernels
- `kernels`: Array of individual kernel data (only present in levels 2 & 4, null in levels 1 & 3)

**Kernel Level (when included):**
- `kernel_name`: Full kernel name
- `time_executions_ms`: Array of execution times across multiple runs
- `is_transformation`: Boolean indicating if kernel is a transpose/transform operation