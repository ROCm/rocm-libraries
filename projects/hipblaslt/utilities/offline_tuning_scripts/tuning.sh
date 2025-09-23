#!/bin/bash

nohup python gemm_tuning.py --input_file example/Qwen3-32B_ali_hipblaslt.log --output_path test_tuning --requested_solution 128 > output.log 2>&1 &
