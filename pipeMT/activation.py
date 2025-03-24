from typing import *

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec

from pipeMT.async_handle import pipeMTAsyncHandle

if TYPE_CHECKING:
    from pipeMT.batch import Batch
    from pipeMT.device import DeviceManager

def async_d2h(compute_stream: torch.cuda.Stream,
              transfer_stream: torch.cuda.Stream,
              transfer_finish_event: Optional[torch.cuda.Event],
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
        if transfer_finish_event is None:
            transfer_stream.synchronize()
        else:
            transfer_finish_event.record()
    return host_tensors

def async_h2d(compute_stream: torch.cuda.Stream,
              transfer_stream: torch.cuda.Stream,
              host_ready_event: Optional[torch.cuda.Event],
              host_tensors: Tuple[Union[torch.Tensor, Any]]
              ) -> List[Union[torch.Tensor, Any]]:
    device_tensors = []
    with torch.cuda.stream(transfer_stream):
        if host_ready_event is not None:
            host_ready_event.wait()
        for host_tensor in host_tensors:
            if isinstance(host_tensor, torch.Tensor):
                device_tensor = host_tensor.to(torch.device('cuda'), non_blocking = True)
                device_tensor.record_stream(compute_stream)
                device_tensors.append(device_tensor)
            else:
                device_tensors.append(host_tensor)
    compute_stream.wait_stream(transfer_stream)
    return device_tensors

class AsyncDownload(torch.autograd.Function):
    @staticmethod
    def forward(ctx, compute_stream: torch.cuda.Stream,
                transfer_stream: torch.cuda.Stream,
                device_tag: torch.Tensor,
                *inputs: Union[torch.Tensor, Any]
                ) -> Tuple[Union[torch.cuda.Event, torch.Tensor, Any], ...]:
        forward_event = torch.cuda.Event()
        backward_event = torch.cuda.Event()
        outputs = async_d2h(compute_stream, transfer_stream, forward_event, inputs)
        ctx.backward_event = backward_event
        ctx.compute_stream = compute_stream
        ctx.transfer_stream = transfer_stream
        return (device_tag, forward_event, backward_event) + tuple(outputs)
    
    @staticmethod
    def backward(ctx, device_tag: torch.Tensor, _, __,
                 *grad_outputs: Union[torch.cuda.Event, Any]
                 ) -> Tuple[Union[None, torch.cuda.Event, Any], ...]:
        backward_event: torch.cuda.Event = ctx.backward_event
        compute_stream: torch.cuda.Stream = ctx.compute_stream
        transfer_stream: torch.cuda.Stream = ctx.transfer_stream
        grad_inputs = async_h2d(compute_stream, transfer_stream, backward_event, grad_outputs)
        return (None, None, device_tag) + tuple(grad_inputs)

class AsyncUpload(torch.autograd.Function):
    @staticmethod
    def forward(ctx, compute_stream: torch.cuda.Stream,
                transfer_stream: torch.cuda.Stream,
                device_tag: torch.Tensor,
                forward_event: torch.cuda.Event,
                backward_event: torch.cuda.Event,
                *inputs: Union[torch.Tensor, Any]
                ) -> Tuple[Union[torch.Tensor, Any], ...]:
        outputs = async_h2d(compute_stream, transfer_stream, forward_event, inputs)
        ctx.backward_event = backward_event
        ctx.compute_stream = compute_stream
        ctx.transfer_stream = transfer_stream
        return (device_tag,) + tuple(outputs)
    
    @staticmethod
    def backward(ctx, device_tag: torch.Tensor,
                 *grad_outputs: Union[torch.cuda.Event, Any]
                 ) -> Tuple[Union[None, torch.cuda.Event, Any], ...]:
        backward_event: torch.cuda.Event = ctx.backward_event
        compute_stream: torch.cuda.Stream = ctx.compute_stream
        transfer_stream: torch.cuda.Stream = ctx.transfer_stream
        grad_inputs = async_d2h(compute_stream, transfer_stream, backward_event, grad_outputs)
        return (None, None, device_tag, None, None) + tuple(grad_inputs)

def download_hidden_state(device: 'DeviceManager', hidden_state
                          ) -> Tuple[Tuple[torch.cuda.Event, ...], Tuple[Any, ...], TreeSpec]:
    flatten_state, flatten_spec = tree_flatten(hidden_state)
    out = AsyncDownload.apply(device.compute_stream, device.activation_downstream,
                              device.order_tag, *flatten_state)
    device.order_tag = out[0]
    return out[1:3], out[3:], flatten_spec

def upload_hidden_state(device: 'DeviceManager', transfer_event: Tuple[torch.cuda.Event, ...],
                        flatten_state: Tuple[Any, ...], flatten_spec: Optional[TreeSpec] = None
                        ) -> Union[Any, List]:
    out = AsyncUpload.apply(device.compute_stream, device.activation_upstream,
                            device.order_tag, *transfer_event, *flatten_state)
    device.order_tag = out[0]
    return list(out[1:]) if flatten_spec is None else tree_unflatten(out[1:], flatten_spec)

def upload_input(device: 'DeviceManager', input: 'Batch', i: int) -> Tuple:
    handle_states = dict()
    for handle in input.input_handles:
        handle_states[handle] = upload_hidden_state(device, handle.transfer_event[i],
                                    handle.flatten_states[i], handle.flatten_specs[i])
    flatten_input_cpu, flatten_spec = tree_flatten((input.input_args[i], input.input_kwargs[i]))
    flatten_input_gpu = upload_hidden_state(device, (None, None), flatten_input_cpu)
    for j in range(len(flatten_input_gpu)):
        if isinstance(flatten_input_gpu[j], pipeMTAsyncHandle):
            flatten_input_gpu[j] = handle_states[flatten_input_gpu[j]]
    return tree_unflatten(flatten_input_gpu, flatten_spec)
