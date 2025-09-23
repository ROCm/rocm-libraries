# hipBLASLt Offline Tuning Script

This tool aims to provide an easy-to-use script to perform GEMM tuning with hipblaslt-bench.

GEMM library [hipBLASLt](https://rocm.docs.amd.com/projects/hipBLASLt/en/latest/index.html) provides benchmark tool for hipBLASLt's supported operations. Please check the [README](https://github.com/ROCm/rocm-libraries/blob/develop/projects/hipblaslt/clients/bench/README.md) for details.

The process of hipBLASLt GEMM tuning includes the followings

1. [Extract hipBLASLt log](#extract-hipblaslt-log)
2. [Pre-Tuning](#pre-tuning)
3. [GEMM Tuning](#gemm-tuning)
4. [Apply Tuning Result](#apply-tuning-result)


## Quick Start
To run GEMM tuning with hipblaslt-bench on Qwen3-32B model.
```bash
nohup python gemm_tuning.py --input_file example/Qwen3-32B_ali_hipblaslt.log --output_path test_tuning --requested_solution 128 > output.log 2>&1 &
```

The tuning generates some output files located at the output path "test_tuning".

        |testing_tuning
        |----> baseline_reproduce_commands.log       # logs that records commands to reproduce baseline result
        |----> tuning_reproduce_commands.log         # logs that records commands to reproduce tuning result
        |----> unique_Qwen3-32B_ali_hipblaslt.log    # hipblaslt log without duplicate lines
        |----> tuning.txt                            # (optional) tuning result, if HIPBLASLT_TUNING_FILE is not defined
        |----> tuning_result.csv                     # includes info of each GEMM and the summary of both baseline and tuning data.


## Extract hipBLASLt log
The hipBLASLt log consists of the hipblaslt-bench command lines for each call of hipBLASLt GEMM kernel. Follow the steps below to extract the hipBLASLt log.

1. Setup Environment Variables
    ```bash
    export HIPBLASLT_LOG_MASK=32
    export HIPBLASLT_LOG_FILE=<path/to/hipblaslt/log>
    ```

2. Run the model

    With the above environment variables, the hipBLASLt log will be automatically saved to the path of the export HIPBLASLT_LOG_MASK=32.

## Pre-Tuning
Before start GEMM tuning, we need to do the following.

1. Setup Environment Variables
    ```bash
    export HIPBLASLT_LOG_MASK=32
    export HIP_VISIBLE_DEVICES=<gpu_id>
    export HIPBLASLT_TUNING_FILE=<path/to/tuning/result>
    ```

2. Remove duplicates in hipBLASLt log

    Since each hipBLASLt GEMM might be called many times when runing the model, the hipBLASLt log contains many duplicate lines. The parse_input_log() in utils.py will remove the duplicate and count the occurance of each GEMM. The hipBLASLt log will be saved under the user-defined output path with name of "unique_<input_hipblaslt_log>".

3. Set-up stable GPU

    A stable GPU frequency is critical for GEMM tuning. The following commands will set-up the GPU frequency relatively stable at 1900 MHz.
    ```bash
    # set-up GPU
    export HIP_FORCE_DEV_KERNARG=1
    rocm-smi --setperfdeterminism 1900 -d <gpu_id>

    # reset GPU
    unset HIP_FORCE_DEV_KERNARG
    rocm-smi -r -d <gpu_id>
    ```


## GEMM Tuning

The GEMM Tuning with hipblaslt-bench is to measure the performance of the candidate kernels within the searching space, and return the kernel with best latency. Runing gemm_tuning.py will start the GEMM tuning with hipblaslt-bench. The options of gemm_tuning.py are listed below.

1. input_file

    The input_file refers to the hipBLASLt log with the GEMMs to be tuned. The duplicate lines in the log will be removed when runing.

2. output_path

    The output_path refers to the directory that user defined to save the tuning result files.

3. swizzleA/B

    Enabling swizzleA/B will re-arrange the memory layout of matrix A/B to avoid bank conflict and boost the latency performance.

4. requested_solution

    The requested_solution define the size of searching space for GEMM tuning. The default value of requested_solution is 128. To measure the baseline latency of the original kernel, just set the request solution to be 1. To expand searching space to the entire kernel libarary, just set the requested solution to be -1.

5. cold_iters & iters

    The cold_iters refers to the warm-up iteration, and the iters refers to the runing iteration to measure kernel latency. The recommended value of cold_iters and iters will be (200, 100) for relative small GEMM, and (10, 2) for relative large GEMM. The default setting will be dynamic iteration, which will adjust the cold_iters and iters based on its size of m*n*k.

6. gpu_id

    The gpu_id determines which GPU you choose to run GEMM tuning. Recommand to use  "rocm-smi" to find a idle GPU to run GEMM tuning for an accurate result.

7. stablize_gpu

    Enable stablize_gpu will provide a lower performance but more consistent GPU setting.


## Apply Tuning Result

After GEMM tuning is done, the tuning result will be saved to the location that HIPBLASLT_TUNING_FILE pointing to. To apply the GEMM tuning result, you need to do the followings.

1. Set-up Environment Variables
    ```bash
    unset HIPBLASLT_TUNING_FILE
    export HIPBLASLT_LOG_MASK=32
    export HIP_VISIBLE_DEVICES=<gpu_id>
    export HIPBLASLT_TUNING_OVERRIDE_FILE=<path/to/tuning/result>
    ```

2. Run the Model

   Run the model again, the tuned GEMM will automatically pick the tuned kernel in the tuning result.


