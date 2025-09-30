import torch.nn.functional as functional;

class BatchnormInference:
    class Input:
        def __init__(self, x: int, mean: int, inv_variance: int, scale: int, bias: int):
            self.x = x
            self.mean = mean
            self.inv_variance = inv_variance
            self.scale = scale
            self.bias = bias
    class Output:
        def __init__(self, y: int):
            self.y = y

    def __init__(self, x: int, mean: int, inv_variance: int, scale: int, bias: int, y: int, name: str = ""):
        self.inputs = BatchnormInference.Input(x, mean, inv_variance, scale, bias)
        self.outputs = BatchnormInference.Output(y)
        self.name = name
        self.type = "BatchnormInferenceAttributes"
    
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
            "type": self.type,
            "name": self.name
        }
    def execute(self):
        inputs = self.inputs
        self.outputs.y.tensor = functional.batch_norm(
            inputs.x.tensor, inputs.mean.tensor, inputs.inv_variance.tensor, inputs.scale.tensor, inputs.bias.tensor, training=False
        )
