import torch;
import torch.nn.functional as functional;
import os
import json;
from typing import NewType

val = torch.float

def invertDict(d):
    return {v: k for (k, v) in d.items()}

class DTypeConverter:
    __dtype_to_string = {
        torch.float : "float",
        torch.float16 : "half",
        torch.bfloat16 : "bfloat16",
        torch.float64 : "double",
        torch.uint8 : "uint8",
        torch.int32: "int32"
    }
    __string_to_dtype = invertDict(__dtype_to_string)
    
    @staticmethod
    def to_string(type: torch.dtype):
        return DTypeConverter.__dtype_to_string[type]
    
    @staticmethod
    def from_string(type: str):
        return DTypeConverter.__string_to_dtype[type]

class Tensor:
    def __init__(self, uid: int, tensor: torch.Tensor, name: str=""):
        self.uid = uid
        self.tensor = tensor
        self.name = name

class TensorAttributes:
    def __init__(self, uid: int, tensor: torch.Tensor, name: str = "", virtual: bool = False):
        self.name = name
        self.uid = uid
        self.strides = tensor.stride()
        self.dims = tensor.size()
        self.data_type = DTypeConverter.to_string(tensor.dtype)
        self.virtual = virtual

class BatchnormInference:
    class Input:
        def __init__(self, 
                     x_tensor_uid: int, 
                     mean_tensor_uid: int, 
                     inv_variance_tensor_uid: int, 
                     scale_tensor_uid: int, 
                     bias_tensor_uid: int):
            self.x_tensor_uid = x_tensor_uid
            self.mean_tensor_uid = mean_tensor_uid
            self.inv_variance_tensor_uid = inv_variance_tensor_uid
            self.scale_tensor_uid = scale_tensor_uid
            self.bias_tensor_uid = bias_tensor_uid
    class Output:
        def __init__(self, y_tensor_uid: int):
            self.y_tensor_uid

    def __init__(self, 
                 x_tensor_uid: int, 
                 mean_tensor_uid: int, 
                 inv_variance_tensor_uid: int, 
                 scale_tensor_uid: int, 
                 bias_tensor_uid: int, 
                 y_tensor_uid: int, 
                 name: str = ""):
        self.input = BatchnormInference.Input(x_tensor_uid, 
                                              mean_tensor_uid, 
                                              inv_variance_tensor_uid, 
                                              scale_tensor_uid, 
                                              bias_tensor_uid)
        self.output = BatchnormInference.Input(y_tensor_uid)
        self.name = name
        self.type = "BatchnormInferenceAttributes"
        

def graph(nodes,
          tensor_data_map: dict[int, torch.Tensor],
          type: torch.dtype = None,
          io_type: torch.dtype = None,
          compute_type: torch.dtype = None,
          intermediate_type: torch.dtype = None,
          name: str=""):
    nodes = [nodes]
    tensors = [TensorAttributes(uid, tensor).__dict__ for (uid, tensor) in tensor_data_map]
    if type != None:
        io_type = compute_type = intermediate_type = type
    elif io_type == None | compute_type == None | intermediate_type == None:
        raise ValueError("type must be set, or io_type, compute_type and intermediate_type must be")

    return {
        "nodes": [node.__dict__ for node in nodes],
        "tensors": tensors,
        "io_type": DTypeConverter.to_string(io_type),
        "compute_type": DTypeConverter.to_string(compute_type),
        "intermediate_type": DTypeConverter.to_string(intermediate_type),
        "name": name
        }

def dump_tensor_data_as_binary(filename: str, tensor: torch.Tensor):
    with open(filename, "wb") as file:
        bytes = bytearray(test.untyped_storage())
        file.write(bytes)

def load_tensor_data_from_binary(filename: str, tensor: torch.Tensor):
    num_bytes = os.path.getsize(filename)
    storage = torch.UntypedStorage.from_file(filename, nbytes=num_bytes)
    tensor.set_(storage, storage_offset=0, size=tuple(tensor.size()), stride=tensor.stride())

def executeBatchnormInference(node: BatchnormInference,  tensor_map: dict[int, torch.Tensor]):
    x = tensor_map[node.input.x_tensor_uid]
    mean = tensor_map[node.input.mean_tensor_uid]
    bias = tensor_map[node.input.bias_tensor_uid]
    inv_variance = tensor_map[node.input.inv_variance_tensor_uid]
    scale = tensor_map[node.input.scale_tensor_uid]

    tensor_map[node.output.y_tensor_id] = functional.batch_norm(input, mean, inv_variance, scale, bias, training=False)

input = torch.tensor([
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

test = torch.tensor([[-1, -2], [-3, -4]], dtype=torch.int16)
print("0 - ", test)
dump_tensor_data_as_binary("test.bin", test)
test.zero_()
print("1 - ", test)
print("size - ", test.size())
print("stride - ", test.stride())

load_tensor_data_from_binary("test.bin", test)
print("2 - ", test)
print("size - ", test.size())
print("stride - ", test.stride())

mean = torch.tensor([1, 2], dtype=torch.float)

print(TensorAttributes(0, input, "x").__dict__)

variance = torch.tensor([3, 4], dtype=torch.float)

bias = torch.tensor([5, 6], dtype=torch.float)

weight = torch.tensor([7, 8], dtype=torch.float)

output = functional.batch_norm(input, mean, variance, weight, bias, training=False)

maps = {0: input, 1: mean, 2: variance, 3: bias, 4: weight, 5: output}

batchnormInferenceNode = BatchnormInference()

print(output)
