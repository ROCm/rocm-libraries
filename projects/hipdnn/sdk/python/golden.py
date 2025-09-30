import os
import torch
from tensor import Tensor, TensorAttributes, dump_data_as_binary, load_data_from_binary
from common import DTypeConverter
from batchnorm_inference import BatchnormInference

        

def graph(nodes,
          tensors: list[Tensor],
          type: torch.dtype = None,
          io_type: torch.dtype = None,
          compute_type: torch.dtype = None,
          intermediate_type: torch.dtype = None,
          name: str=""):
    tensors = [TensorAttributes(tensor).__dict__ for tensor in tensors]
    if type != None:
        io_type = compute_type = intermediate_type = type
    elif io_type == None | compute_type == None | intermediate_type == None:
        raise ValueError("type must be set, or io_type, compute_type and intermediate_type must be")

    return {
        "nodes": [node.as_dict() for node in nodes],
        "tensors": tensors,
        "io_type": DTypeConverter.to_string(io_type),
        "compute_type": DTypeConverter.to_string(compute_type),
        "intermediate_type": DTypeConverter.to_string(intermediate_type),
        "name": name
        }

input = Tensor(torch.tensor([
        [
            [
                [ 0.,  1.], 
                [ 2.,  3.]
            ],
            [
                [ 4.,  5.],
                [ 6.,  7.]
            ]
        ],
               [
            [
                [ 0.,  -1.], 
                [ -2.,  -3.]
            ],
            [
                [ -4.,  -5.],
                [ -6.,  -7.]
            ]
        ]
    ], dtype=torch.float)
)

test = torch.tensor([[-1, -2], [-3, -4]], dtype=torch.int16)
print("0 - ", test)
dump_data_as_binary("test.bin", test)
test.zero_()
print("1 - ", test)
print("size - ", test.size())
print("stride - ", test.stride())

load_data_from_binary("test.bin", test)
print("2 - ", test)
print("size - ", test.size())
print("stride - ", test.stride())

mean = Tensor(torch.tensor([1, 2], dtype=torch.float))

print(TensorAttributes(input).__dict__)

inv_variance = Tensor(torch.tensor([3, 4], dtype=torch.float))

bias = Tensor(torch.tensor([5, 6], dtype=torch.float))

scale = Tensor(torch.tensor([7, 8], dtype=torch.float))

output = Tensor(torch.tensor([]))

# output = functional.batch_norm(input, mean, inv_variance, scale, bias, training=False)

batchnormInferenceNode = BatchnormInference(input, mean, inv_variance, scale, bias, output)
batchnormInferenceNode.execute()

print(output.tensor)

print(graph([batchnormInferenceNode], [input, mean, inv_variance, scale, bias, output], type=torch.float, name="Graph"))
