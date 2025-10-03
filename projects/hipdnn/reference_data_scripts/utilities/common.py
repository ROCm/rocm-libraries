import torch

NODE_REGISTRY = {}

def register_node(node_class):
    NODE_REGISTRY[node_class.type_str] = node_class
    return node_class


class IncrementalId:
    def __init__(self):
        self.__id = -1
    def get(self):
        self.__id += 1
        return self.__id

tensor_uid = IncrementalId()

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
