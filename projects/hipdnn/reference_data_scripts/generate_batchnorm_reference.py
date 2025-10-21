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

def save_batchnorm_inference_execution(x_size: list[int], 
                                       io_type: torch.dtype, 
                                       intermediate_type: torch.dtype, 
                                       min_val, 
                                       max_val, 
                                       base_filename: str,
                                       using_gpu: bool):
    channel_idx = 1
    derived_sizes = [1 for _ in x_size]
    derived_sizes[channel_idx] = x_size[channel_idx]

    x            = TensorAttributes.random(min_val, max_val, io_type, x_size)
    mean         = TensorAttributes.random(min_val, max_val, intermediate_type, derived_sizes)
    inv_variance = TensorAttributes.random(min_val, max_val, intermediate_type, derived_sizes)
    scale        = TensorAttributes.random(min_val, max_val, intermediate_type, derived_sizes)
    bias         = TensorAttributes.random(min_val, max_val, intermediate_type, derived_sizes)
    y            = TensorAttributes.empty()

    node = BatchnormInference(x, mean, inv_variance, scale, bias, y)

    node.execute(using_gpu)

    graph = Graph([node], io_type=io_type, compute_type=io_type, intermediate_type=intermediate_type)
    graph.save(base_filename)

def main():
    parser = argparse.ArgumentParser(
        prog="generate_batchnorm_reference", 
        description="Executes batchnorm problem in pytorch, and saves the execution data in a hipDNN readable format"
    )
    parser.add_argument("-f", "--base-filename", required=True, type=str,
                        help="base file name and path for output files (without extensions)")
    parser.add_argument("-d", "--io-type", required=True, type=str, 
                        help="datatype for batch norm operations (float, half, bfloat16, double, uint8, int32)")
    parser.add_argument("-s", "--size", required=True, nargs="+", type=int, 
                        help="size of x tensor (ex: --size 2 4 6 8)")
    parser.add_argument("--min", default=0.1, type=float, help="minimum value in tensor")
    parser.add_argument("--max", default=1.0, type=float, help="maximum value in tensor")
    parser.add_argument("--seed", default=0, type=int, help="seed for random data in tensors")
    parser.add_argument("--gpu", default=False, type=bool, help="Use cuda or rocm backend")

    args = parser.parse_args()

    if args.gpu and not torch.cuda.is_available():
        return RuntimeError("GPU backend not available")
    elif args.gpu:
        print("Executing graph on", torch.cuda.get_device_name())

    torch.manual_seed(args.seed)
    save_batchnorm_inference_execution(args.size, 
                                       DTypeConverter.from_string(args.io_type), 
                                       torch.float, 
                                       args.min, 
                                       args.max, 
                                       args.base_filename,
                                       args.gpu)

if __name__ == "__main__":
    main()

