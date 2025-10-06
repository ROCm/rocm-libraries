import torch
from tensor import TensorAttributes, dump_data_as_binary, load_data_from_binary
from common import DTypeConverter, NODE_REGISTRY
from batchnorm_inference import BatchnormInference
from itertools import chain
import json
import os

def node_from_dict(node_dict: dict, tensors: dict[int, TensorAttributes]):
    if node_dict["type"] not in NODE_REGISTRY.keys():
        raise RuntimeError("Unsupported node type: "+ node_dict["type"])

    return NODE_REGISTRY[node_dict["type"]].from_dict(node_dict, tensors)

class Graph:
    def  __init__(self, nodes,
                  dtype: torch.dtype = None,
                  io_type: torch.dtype = None,
                  compute_type: torch.dtype = None,
                  intermediate_type: torch.dtype = None,
                  name: str=""):
        if dtype != None and (io_type == None and compute_type == None and intermediate_type == None):
            io_type = compute_type = intermediate_type = dtype
        elif dtype != None:
            raise ValueError("type must be set, or all of io_type, compute_type and intermediate_type must be")

        self.nodes = nodes
        self.tensors = dict()
        for node in nodes:
            inputs = node.inputs.__dict__.values()
            outputs = node.outputs.__dict__.values()
            tensors = filter(lambda x: type(x) == TensorAttributes, chain(inputs, outputs))
            self.tensors.update({ tensor.uid: tensor for tensor in tensors })

        self.io_type = io_type
        self.compute_type = compute_type
        self.intermediate_type = intermediate_type
        self.name = name

    # TODO Implement execute for more complicated graphs
    # def execute():

    def as_dict(self):
        tensors = [tensor.as_dict() for tensor in self.tensors.values()]
        return {
            "nodes": [node.as_dict() for node in self.nodes],
            "tensors": tensors,
            "io_type": DTypeConverter.to_string(self.io_type),
            "compute_type": DTypeConverter.to_string(self.compute_type),
            "intermediate_type": DTypeConverter.to_string(self.intermediate_type),
            "name": self.name
        }

    def save(self, base_filename: str):
        with open(base_filename+".json", "w") as file:
            json.dump(self.as_dict(), file)
        for tensor in self.tensors.values():
            dump_data_as_binary("{}.tensor{}.bin".format(base_filename, tensor.uid), tensor)

    @staticmethod
    def from_file(base_path: str):
        with open(base_path+".json") as file:
            d = json.load(file)

        tensor_list = [TensorAttributes.from_dict(tensorDict) for tensorDict in d["tensors"]]
        tensor_map = {tensor.uid: tensor for tensor in tensor_list}

        nodes = [node_from_dict(node_dict, tensor_map) for node_dict in d["nodes"]]

        for (uid, tensor) in tensor_map.items():
            filepath = "{}.tensor{}".format(base_path, uid)
            if os.path.exists(filepath):
                load_data_from_binary(filepath, tensor)

        io_type = DTypeConverter.from_string(d["io_type"])
        compute_type = DTypeConverter.from_string(d["compute_type"])
        intermediate_type = DTypeConverter.from_string(d["intermediate_type"])

        return Graph(nodes, io_type=io_type, compute_type=compute_type, intermediate_type=intermediate_type, name=d["name"] )



