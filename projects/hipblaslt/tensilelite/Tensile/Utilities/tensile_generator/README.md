## Tensile Config Generator (Beta)

The `tensile_config_generator.py` script is a tool designed to generate Tensile configuration files for hipBLASLt. These configuration files are essential for optimizing GEMM (General Matrix Multiplication) operations on AMD GPUs.

### Purpose

The main purpose of this script is to:
1. Analyze the hipBLASLt logs for GEMM operations
2. Generate optimized Tensile configurations based on the identified GEMM patterns
3. Produce YAML files that can be used by hipBLASLt to tune GEMM operations

### Usage

To use the `tensile_config_generator.py` script, follow these steps:

1. Generate hipBLASLt logs:
   ```
   HIPBLASLT_LOG_MASK=32 HIPBLASLT_LOG_FILE=./hipblaslt_%i.log <CMD>
   ```

2. Run the script from the command line:
   ```
   python ./tensile_config_generator.py [options]
   ```

   Available options:

   | Option | Description |
   |--------|-------------|
   | `-h, --help` | Show this help message and exit |
   | `--hipblaslt_log HIPBLASLT_LOG` | Path to hipblaslt log file |
   | `--tensile_config TENSILE_CONFIG` | Path to tensile config file |
   | `--gpus GPUS` | Number of GPUs for tuning hipblaslt |
   | `--topk TOPK` | Top k GEMMs for tuning |
   | `--iters ITERS` | Max tuning iterations |
   | `--fast BOOL` | If enabled, only tune the matrix instruction with min tile sizes, else, tune full matrix instructions |
   | `--gridbase_config GRIDBASE_CONFIG` | Path to gridbase config file |
   | `--full_mfma BOOL` | If enabled, will search for all mfma instructions |
   | `--full_stage BOOL` | If enabled, will search for all stages instructions |
   | `--num_stages STAGES` | How many times to divide matrix |

   Equality tuning example:
   ```
   python ./tensile_config_generator.py --hipblaslt_log ./hipblaslt_gemm_log_example.txt --tensile_config ./tuning_template.yaml --gpus 4 --iters 100
   ```

   Gridbase tuning example:
   ```
   python ./tensile_config_generator.py --gridbase_config ./gridbase_config_template.yaml --tensile_config ./tuning_template.yaml --gpus 4 --iters 100
   ```

3. Install hipBLASLt and Tensile (change the path to the hipBLASLt repo):
   ```
   bash ./install.sh -idc -a $(/opt/rocm/llvm/bin/offload-arch) --cpu_ref_lib=lapack
   ```

4. Tune GEMM kernels using the generated YAML files:
   ```
   HIP_FORCE_DEV_KERNARG=1 ./tensilelite/Tensile/bin/Tensile <generated yaml path> <tune result directory>
   ```

5. Merge tune results:

   gfx942:

   ```
   python3 ./tensilelite/Tensile/Utilities/merge.py --no_eff library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aquavanjaram/gfx942_{cu count}/{Equality|GridBased}/ <tune result directory>/3_LibraryLogic/ library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aquavanjaram/gfx942_{cu count}/{Equality|GridBased}/
   ```
   gfx90a:

   ```
   python3 ./tensilelite/Tensile/Utilities/merge.py --no_eff library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aldebaran/{cu count}/{Equality|GridBased}/ <tune result directory>/3_LibraryLogic/ library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/aldebaran/{cu count}/{Equality|GridBased}/
   ```

6. Rebuild hipBLASLt with the merged results:
   ```
   bash ./install.sh -idc -a $(/opt/rocm/llvm/bin/offload-arch) --cpu_ref_lib=lapack
   ```

For more detailed information on the script's functionality and advanced usage, please refer to the comments within the `tensile_config_generator.py` file.
