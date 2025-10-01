import torch
from tensor import TensorAttributes, dump_data_as_binary
from common import DTypeConverter
from batchnorm_inference import BatchnormInference
import json


def graph(nodes,
          tensors: list[TensorAttributes],
          io_type: torch.dtype = None,
          compute_type: torch.dtype = None,
          intermediate_type: torch.dtype = None,
          name: str=""):
    tensors = [tensor.as_dict() for tensor in tensors]

    return {
        "nodes": [node.as_dict() for node in nodes],
        "tensors": tensors,
        "io_type": DTypeConverter.to_string(io_type),
        "compute_type": DTypeConverter.to_string(compute_type),
        "intermediate_type": DTypeConverter.to_string(intermediate_type),
        "name": name
        }

def save_graph(base_filename: str, nodes, tensors: list[TensorAttributes], type: torch.dtype = None,
          io_type: torch.dtype = None,
          compute_type: torch.dtype = None,
          intermediate_type: torch.dtype = None,
          graph_name: str=""):
    if type != None and (io_type == None and compute_type == None and intermediate_type == None):
        io_type = compute_type = intermediate_type = type
    elif type != None:
        raise ValueError("type must be set, or all of io_type, compute_type and intermediate_type must be")

    graphDict = graph(nodes, tensors, io_type, compute_type, intermediate_type, graph_name)
    with open(base_filename+".json", "w") as file:
        json.dump(graphDict, file)
    for tensor in tensors:
        dump_data_as_binary("{}.tensor{}.bin".format(base_filename, tensor.uid), tensor.tensor)
