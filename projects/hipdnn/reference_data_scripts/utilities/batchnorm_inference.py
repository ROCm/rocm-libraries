import torch.nn.functional as functional;
from tensor import TensorAttributes
from common import register_node

@register_node
class BatchnormInference:
    type_str = "BatchnormInferenceAttributes"
    class Input:
        def __init__(self,
                     x: TensorAttributes,
                     mean: TensorAttributes,
                     inv_variance: TensorAttributes,
                     scale: TensorAttributes,
                     bias: TensorAttributes):
            self.x = x
            self.mean = mean
            self.inv_variance = inv_variance
            self.scale = scale
            self.bias = bias
    class Output:
        def __init__(self, y: TensorAttributes):
            self.y = y

    def __init__(self,
                 x: TensorAttributes,
                 mean: TensorAttributes,
                 inv_variance: TensorAttributes,
                 scale: TensorAttributes,
                 bias: TensorAttributes,
                 y: TensorAttributes,
                 name: str = ""):
        self.inputs = BatchnormInference.Input(x, mean, inv_variance, scale, bias)
        self.outputs = BatchnormInference.Output(y)
        self.name = name

    def as_dict(self):
        return {
            "inputs":{
                "x_tensor_uid": self.inputs.x.uid,
                "mean_tensor_uid": self.inputs.mean.uid,
                "inv_variance_tensor_uid": self.inputs.inv_variance.uid,
                "scale_tensor_uid": self.inputs.scale.uid,
                "bias_tensor_uid": self.inputs.bias.uid,
            },
            "outputs":{
                "y_tensor_uid": self.outputs.y.uid
            },
            "type": BatchnormInference.type_str,
            "name": self.name
        }
    def execute(self, using_gpu: bool):
        inputs = self.inputs

        # pytorch and hipdnn disagree on the dimension of mean, inv_variance, etc...
        # hipdnn requires the same dimension as x, with 1's in every dimension aside from the channel
        # pytorch requires they have a dimension of 1 with the size being the number of channels
        original_dims = inputs.mean.tensor.size()
        new_dims = [original_dims[1]]
        
        inputs.mean.tensor.resize_(new_dims)
        inputs.scale.tensor.resize_(new_dims)
        inputs.bias.tensor.resize_(new_dims)
        inputs.inv_variance.tensor.resize_(new_dims)

        if using_gpu:
            inputs.x.to_gpu()
            inputs.mean.to_gpu()
            inputs.inv_variance.to_gpu()
            inputs.scale.to_gpu()
            inputs.bias.to_gpu()


        saved_exception = None
        
        try:
            self.outputs.y.tensor = functional.batch_norm(
                inputs.x.tensor,
                inputs.mean.tensor,
                inputs.inv_variance.tensor,
                inputs.scale.tensor,
                inputs.bias.tensor,
                training=False
            )
        except Exception as e:
            saved_exception = e
        finally:
            # Restore to original sizes
            inputs.mean.tensor.resize_(original_dims)
            inputs.scale.tensor.resize_(original_dims)
            inputs.bias.tensor.resize_(original_dims)
            inputs.inv_variance.tensor.resize_(original_dims)

            if using_gpu:
                inputs.x.to_cpu()
                inputs.mean.to_cpu()
                inputs.inv_variance.to_cpu()
                inputs.scale.to_cpu()
                inputs.bias.to_cpu()


        if saved_exception is not None:
            raise saved_exception

    @staticmethod
    def from_dict(d: dict, tensors: dict[int, TensorAttributes]):
        inputs = d["inputs"]
        outputs = d["outputs"]
        
        return BatchnormInference(tensors[inputs["x_tensor_uid"]],
                                  tensors[inputs["mean_tensor_uid"]],
                                  tensors[inputs["inv_variance_tensor_uid"]],
                                  tensors[inputs["scale_tensor_uid"]],
                                  tensors[inputs["bias_tensor_uid"]],
                                  tensors[outputs["y_tensor_uid"]],
                                  d["name"])
