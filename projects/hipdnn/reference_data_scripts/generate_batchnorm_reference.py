import sys
import os
# Find files in utilities folder without requiring a package to be created
sys.path.insert(1, os.path.dirname(os.path.realpath(__file__))+"/utilities/")
import torch
from tensor import TensorAttributes
from batchnorm_inference import BatchnormInference
from graph import Graph
from common import DTypeConverter
import argparse

def save_batchnorm_inference_execution(x_size: list[int], dtype: torch.dtype, min_val, max_val, base_filename: str):
    channel_idx = 1
    derived_sizes = [1, x_size[channel_idx], 1, 1]

    x            = TensorAttributes.random(min_val, max_val, dtype, x_size)
    mean         = TensorAttributes.random(min_val, max_val, dtype, derived_sizes)
    inv_variance = TensorAttributes.random(min_val, max_val, dtype, derived_sizes)
    scale        = TensorAttributes.random(min_val, max_val, dtype, derived_sizes)
    bias         = TensorAttributes.random(min_val, max_val, dtype, derived_sizes)
    y            = TensorAttributes.empty()

    node = BatchnormInference(x, mean, inv_variance, scale, bias, y)

    node.execute()

    graph = Graph([node], dtype)
    graph.save(base_filename)

def main():
    parser = argparse.ArgumentParser(
        prog="generate_batchnorm_reference", 
        description="Executes batchnorm problem in pytorch, and saves the execution data in a hipDNN readable format"
    )
    parser.add_argument("-f", "--base-filename", required=True, type=str,
                        help="base file name and path for output files (without extensions)")
    parser.add_argument("-d", "--datatype", required=True, type=str, 
                        help="datatype for batch norm operations (float, half, bfloat16, double, uint8, int32)")
    parser.add_argument("-s", "--size", required=True, nargs="+", type=int, 
                        help="size of x tensor (ex: --size 2 4 6 8)")
    parser.add_argument("--min", default=-1.0, type=float, help="minimum value in tensor")
    parser.add_argument("--max", default=1.0, type=float, help="maximum value in tensor")
    parser.add_argument("--seed", default=0, type=int, help="seed for random data in tensors")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    save_batchnorm_inference_execution(args.size, DTypeConverter.from_string(args.datatype), args.min, args.max, args.base_filename)

if __name__ == "__main__":
    main()

