import torch
import os


def make_directories(dir_path, model_name):
    if not os.path.exists(f"{dir_path}/{model_name}/"):
        os.makedirs(f"{dir_path}/{model_name}/")


def export_rnnt_onnx(model, x, dir_path):

    for component in ['rnnt_encoder', 'rnnt_prediction', 'rnnt_joint']:
        make_directories(dir_path, component)

    seq_length, batch_size, feature_length = x.shape
    inp = torch.randn([seq_length, batch_size, feature_length]).to('cuda')
    feature_length = torch.LongTensor([seq_length]).to('cuda')
    x_padded, x_lens = model.encoder(inp, feature_length)
    torch.onnx.export(model.encoder, (inp, feature_length),
                      f"{dir_path}/rnnt_encoder/model.onnx",
                      opset_version=12,
                      input_names=['input', 'feature_length'],
                      output_names=['x_padded', 'x_lens'],
                      dynamic_axes={'input': {
                          0: 'seq_len',
                          1: 'batch'
                      }})

    symbol = torch.LongTensor([[20]]).to('cuda')
    hidden = torch.randn([2, batch_size,
                          320]).to('cuda'), torch.randn([2, batch_size,
                                                         320]).to('cuda')
    g, hidden = model.prediction.forward(symbol, hidden)
    torch.onnx.export(model.prediction, (symbol, hidden),
                      f"{dir_path}/rnnt_prediction/model.onnx",
                      opset_version=12,
                      input_names=['symbol', 'hidden_in_1', 'hidden_in_2'],
                      output_names=['g', 'hidden_out_1', 'hidden_out_2'],
                      dynamic_axes={
                          'symbol': {
                              0: 'batch'
                          },
                          'hidden_in_1': {
                              1: 'batch'
                          },
                          'hidden_in_2': {
                              1: 'batch'
                          }
                      })

    f = torch.randn([batch_size, 1, 1024]).to('cuda')
    model.joint.forward(f, g).to('cuda')
    torch.onnx.export(model.joint, (f, g),
                      f"{dir_path}/rnnt_joint/model.onnx",
                      opset_version=12,
                      input_names=['0', '1'],
                      output_names=['result'],
                      dynamic_axes={
                          '0': {
                              0: 'batch'
                          },
                          '1': {
                              0: 'batch'
                          }
                      })
