from typing import *

import torch

def upload_layer(layer: torch.nn.Module, transfer_stream: torch.cuda.Stream,
                 compute_stream: torch.cuda.Stream):
    for param in layer.parameters():
        with torch.cuda.stream(transfer_stream):
            param.data = param.data_cpu.to(device = torch.cuda.current_device(), non_blocking = True)
            param.data.record_stream(compute_stream)
        param.grad = None
        param.transfer_stream = transfer_stream
    for buffer in layer.buffers():
        with torch.cuda.stream(transfer_stream):
            buffer.data = buffer.data_cpu.to(device = torch.cuda.current_device(), non_blocking = True)
            buffer.data.record_stream(compute_stream)
    
def free_layer_gpu(layer: torch.nn.Module, transfer_stream: torch.cuda.Stream):
    return
    with torch.cuda.stream(transfer_stream):
        for buffer in layer.buffers():
            buffer.data.record_stream(transfer_stream)
            buffer.data_cpu.copy_(buffer.data, non_blocking = True)
            buffer.data = buffer.data_cpu

def download_grad(param: torch.Tensor):
    param.data = param.data_cpu
    transfer_stream: torch.cuda.Stream = param.transfer_stream
    with torch.cuda.stream(transfer_stream):
        param.grad.record_stream(transfer_stream)
        param.grad = param.grad.to(torch.device('cpu'), non_blocking = True)

def preprocess_param(model: torch.nn.Module):
    model._apply(lambda t: t.pin_memory())
    for parm in model.parameters():
        parm.data_cpu = parm.data
        parm.register_post_accumulate_grad_hook(download_grad)
    for buffer in model.buffers():
        buffer.data_cpu = buffer.data
