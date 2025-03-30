from typing import *

import torch

def async_d2h(compute_stream: torch.cuda.Stream,
              transfer_stream: torch.cuda.Stream,
              transfer_finish_event: List[Optional[torch.cuda.Event]],
              device_tensors: Tuple[Union[torch.Tensor, Any]]
              ) -> List[Union[torch.Tensor, Any]]:
    host_tensors = []
    transfer_stream.wait_stream(compute_stream)
    with torch.cuda.stream(transfer_stream):
        for device_tensor in device_tensors:
            if isinstance(device_tensor, torch.Tensor):
                device_tensor.record_stream(transfer_stream)
                host_tensors.append(device_tensor.to(torch.device('cpu'), non_blocking = True))
            else:
                host_tensors.append(device_tensor)
        for event in transfer_finish_event:
            if event is None:
                transfer_stream.synchronize()
            else:
                event.record()
    return host_tensors

def async_h2d(compute_stream: torch.cuda.Stream,
              transfer_stream: torch.cuda.Stream,
              host_ready_event: List[torch.cuda.Event],
              host_tensors: Tuple[Union[torch.Tensor, Any]]
              ) -> List[Union[torch.Tensor, Any]]:
    device_tensors = []
    with torch.cuda.stream(transfer_stream):
        for event in host_ready_event:
            event.wait()
        for host_tensor in host_tensors:
            if isinstance(host_tensor, torch.Tensor):
                device_tensor = host_tensor.to(transfer_stream.device, non_blocking = True)
                device_tensor.record_stream(compute_stream)
                device_tensors.append(device_tensor)
            else:
                device_tensors.append(host_tensor)
    compute_stream.wait_stream(transfer_stream)
    return device_tensors

def upload_layer(layer: torch.nn.Module, transfer_stream: torch.cuda.Stream,
                 compute_stream: torch.cuda.Stream, upload_grad: bool):
    with torch.cuda.stream(transfer_stream):
        for param in layer.parameters():
            param.data = param.data_cpu.to(transfer_stream.device, non_blocking = True)
            param.data.record_stream(compute_stream)
            if upload_grad and param.grad is not None:
                param.grad = param.grad.to(transfer_stream.device, non_blocking = True)
                param.grad.record_stream(compute_stream)
        for buffer in layer.buffers():
            buffer.data = buffer.data_cpu.to(transfer_stream.device, non_blocking = True)
            buffer.data.record_stream(compute_stream)

def free_layer(layer: torch.nn.Module):
    for param in layer.parameters():
        param.data = param.data_cpu
    for buffer in layer.buffers():
        buffer.data = buffer.data_cpu

def download_layer(layer: torch.nn.Module, transfer_stream: torch.cuda.Stream):
    with torch.cuda.stream(transfer_stream):
        for param in layer.parameters():
            param.data = param.data_cpu
            param.grad.record_stream(transfer_stream)
            param.grad = param.grad.to(torch.device('cpu'), non_blocking = True)
        for buffer in layer.buffers():
            buffer.data.record_stream(transfer_stream)
            buffer.data_cpu.copy_(buffer.data, non_blocking = True)
            buffer.data = buffer.data_cpu
