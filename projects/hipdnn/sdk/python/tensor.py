import torch
from common import DTypeConverter, tensor_uid
import os

class Tensor:
    def __init__(self, tensor: torch.Tensor, uid: int=None, name: str="", virtual: bool = False):
        self.uid = uid if uid != None else tensor_uid.get() 
        self.tensor = tensor
        self.name = name
        self.virtual = virtual
    def __eq__(self, other):
        return self.uid == other.uid
    def __hash__(self):
        return self.uid
        
class TensorAttributes:
    def __init__(self, tensor: Tensor):
        self.name = tensor.name
        self.uid = tensor.uid
        self.strides = list(tensor.tensor.stride())
        self.dims = list(tensor.tensor.size())
        self.data_type = DTypeConverter.to_string(tensor.tensor.dtype)
        self.virtual = tensor.virtual

def dump_data_as_binary(filename: str, tensor: torch.Tensor):
    with open(filename, "wb") as file:
        bytes = bytearray(tensor.untyped_storage())
        file.write(bytes)

def load_data_from_binary(filename: str, tensor: torch.Tensor):
    num_bytes = os.path.getsize(filename)
    storage = torch.UntypedStorage.from_file(filename, nbytes=num_bytes)
    tensor.set_(storage, storage_offset=0, size=tuple(tensor.size()), stride=tensor.stride())
