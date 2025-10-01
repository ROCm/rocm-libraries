import torch
from tensor import TensorAttributes
from batchnorm_inference import BatchnormInference
from graph import save_graph
import sys

def save_batchnorm_inference_execution(x_size: list[int], dtype: torch.dtype, min_val, max_val, base_filename: str):
    other_sizes = [x_size[1]]

    x            = TensorAttributes.random(min_val, max_val, dtype, x_size)
    mean         = TensorAttributes.random(min_val, max_val, dtype, other_sizes)
    inv_variance = TensorAttributes.random(min_val, max_val, dtype, other_sizes)
    scale        = TensorAttributes.random(min_val, max_val, dtype, other_sizes)
    bias         = TensorAttributes.random(min_val, max_val, dtype, other_sizes)
    y            = TensorAttributes.empty()

    node = BatchnormInference(x, mean, inv_variance, scale, bias, y)

    node.execute()

    modified_filename = "{}_{}_{}_to_{}".format(base_filename, dtype, min_val, max_val)

    save_graph(modified_filename, [node], [x, mean, inv_variance, scale, bias, y], dtype)


def main(args):
    if len(args) == 0:
        print("Usage: batchnorm_reference.py <destination_path>")
        exit()

    torch.manual_seed(121)
    base_file_path = args[0] + "BatchnormInferencePytorchRef"


    save_batchnorm_inference_execution([3, 7, 100, 100], torch.float,    -100.0, 100.0, base_file_path)
    save_batchnorm_inference_execution([3, 7, 100, 100], torch.half,     -100.0, 100.0, base_file_path)
    save_batchnorm_inference_execution([3, 7, 100, 100], torch.bfloat16, -100.0, 100.0, base_file_path)

if __name__ == "__main__":
    main(sys.argv[1:])

