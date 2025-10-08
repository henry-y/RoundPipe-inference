from typing import *

import torch

def async_d2h(compute_stream: torch.cuda.Stream,
              transfer_stream: torch.cuda.Stream,
              transfer_finish_event: Iterable[Optional[torch.cuda.Event]],
              device_tensors: Iterable[Union[torch.Tensor, Any]]
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
              host_ready_event: Iterable[torch.cuda.Event],
              host_tensors: Iterable[Union[torch.Tensor, Any]]
              ) -> List[Union[torch.Tensor, Any]]:
    device_tensors = []
    with torch.cuda.stream(transfer_stream):
        for event in host_ready_event:
            event.wait()
        for host_tensor in host_tensors:
            if isinstance(host_tensor, torch.Tensor):
                try:
                    pinned_host_tensor = host_tensor.pin_memory()
                except RuntimeError:
                    pinned_host_tensor = host_tensor.clone().pin_memory()
                device_tensor = pinned_host_tensor.to(transfer_stream.device, non_blocking = True)
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
            # 懒加载：若 CPU 侧为空，占位符触发从 HF 分片读取
            if getattr(param, 'data_cpu', None) is None:
                pipe_ref = getattr(layer, '_pipeMT_ref', None)
                if pipe_ref is None or pipe_ref.hf_loader is None:
                    raise RuntimeError('param.data_cpu is None but no hf_loader available')
                name = getattr(param, '_name', None)
                if name is None:
                    raise RuntimeError('Parameter missing _name for hf lazy load')
                param.data_cpu = pipe_ref.hf_loader.load_to_cpu(name, param.dtype, param.shape)
            param.data = param.data_cpu.to(transfer_stream.device, non_blocking = True) # type: ignore[attr-defined]
            param.data.record_stream(compute_stream)
            if upload_grad and param.grad is not None:
                param.grad = param.grad.to(transfer_stream.device, non_blocking = True)
                param.grad.record_stream(compute_stream)
        for buffer in layer.buffers():
            if getattr(buffer, 'data_cpu', None) is None:
                pipe_ref = getattr(layer, '_pipeMT_ref', None)
                if pipe_ref is None or pipe_ref.hf_loader is None:
                    raise RuntimeError('buffer.data_cpu is None but no hf_loader available')
                name = getattr(buffer, '_name', None)
                if name is None:
                    raise RuntimeError('Buffer missing _name for hf lazy load')
                buffer.data_cpu = pipe_ref.hf_loader.load_to_cpu(name, buffer.dtype, buffer.shape)
            buffer.data = buffer.data_cpu.to(transfer_stream.device, non_blocking = True) # type: ignore[attr-defined]
            buffer.data.record_stream(compute_stream)

def free_layer(layer: torch.nn.Module):
    for param in layer.parameters():
        param.data = param.data_cpu # type: ignore[attr-defined]
    for buffer in layer.buffers():
        buffer.data = buffer.data_cpu # type: ignore[attr-defined]

def download_layer(layer: torch.nn.Module, transfer_stream: torch.cuda.Stream):
    with torch.cuda.stream(transfer_stream):
        for param in layer.parameters():
            param.data = param.data_cpu # type: ignore[attr-defined]
            param.grad.record_stream(transfer_stream) # type: ignore[union-attr]
            param.grad = param.grad.to(torch.device('cpu'), non_blocking = True) # type: ignore[union-attr]
        for buffer in layer.buffers():
            buffer.data.record_stream(transfer_stream)
            buffer.data_cpu.copy_(buffer.data, non_blocking = True) # type: ignore[attr-defined]
            buffer.data = buffer.data_cpu # type: ignore[attr-defined]

class PinnedUpload(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t: torch.Tensor, d: torch.device):
        return t.pin_memory().to(d, non_blocking = True)
    
    @staticmethod
    def backward(ctx, g: torch.Tensor): # type: ignore[override]
        g_host = torch.empty_like(g, device = torch.device('cpu'), pin_memory = True)
        g_host.copy_(g)
        return g_host, None
