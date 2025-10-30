import torch
from common import DTypeConverter, tensor_uid
import os

class TensorAttributes:
    def __init__(self, tensor: torch.Tensor, uid: int=None, name: str="", virtual: bool = False):
        self.uid = uid if uid != None else tensor_uid.get()
        self.tensor = tensor
        self.name = name
        self.virtual = virtual
    def __eq__(self, other):
        return self.uid == other.uid
    def __hash__(self):
        return self.uid
    def as_dict(self):
        return {
            "name": self.name,
            "uid": self.uid,
            "strides": list(self.tensor.stride()),
            "dims": list(self.tensor.size()),
            "data_type": DTypeConverter.to_string(self.tensor.dtype),
            "virtual": self.virtual
        }

    def to_gpu(self):
        tensor = self.tensor.cuda()

    def to_cpu(self):
        tensor = self.tensor.cpu()

    @staticmethod
    def empty(dtype: torch.dtype=torch.float, dims: list[int]=[], uid: int=None, name: str="", virtual: bool=False):
        return TensorAttributes(torch.empty(dims, dtype=dtype), uid, name, virtual)

    @staticmethod
    def random(min_val, max_val, dtype: torch.dtype, dims: list[int], generator: torch.Generator = None):
        if (dtype == torch.int or dtype == torch.int8 or dtype == torch.int16
            or dtype == torch.int32 or dtype == torch.int64):
            return TensorAttributes(torch.randint(low=int(min_val), high=int(max_val), dtype=dtype, size=dims, generator=generator))

        tensor = torch.rand(dtype=dtype, size=dims, generator=generator)
        return TensorAttributes(tensor*(max_val - min_val) + min_val)

    @staticmethod
    def from_dict(d: dict):
        dtype = DTypeConverter.from_string(d["data_type"])
        return TensorAttributes.empty(dtype, d["dims"], d["uid"], d["name"], d["virtual"])



def dump_data_as_binary(filename: str, tensor_attr: TensorAttributes):
    tensor = tensor_attr.tensor
    with open(filename, "wb") as file:
        bytes = bytearray(tensor.untyped_storage())
        file.write(bytes)

def load_data_from_binary(filename: str, tensor_attr: TensorAttributes):
    tensor = tensor_attr.tensor
    num_bytes = os.path.getsize(filename)
    storage = torch.UntypedStorage.from_file(filename, nbytes=num_bytes)
    tensor.set_(storage, storage_offset=0, size=tuple(tensor.size()), stride=tensor.stride())
