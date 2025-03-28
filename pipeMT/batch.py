from typing import *
import warnings

import torch
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks, merge_chunks
from torch.utils._pytree import tree_flatten, tree_unflatten, TreeSpec

from pipeMT.async_handle import pipeMTAsyncHandle

if TYPE_CHECKING:
    from torch.distributed.pipelining.microbatch import TensorChunkSpec

class Batch:
    def __init__(self, *args,
                 num_microbatch: Optional[int] = torch.cuda.device_count() + 1,
                 args_chunk_spec: Optional[Tuple['TensorChunkSpec', ...]] = None,
                 kwargs_chunk_spec: Optional[Dict[str, 'TensorChunkSpec']] = None,
                 result_chunk_spec: Optional[Tuple['TensorChunkSpec', ...]] = None,
                 **kwargs):
        self.input_handles: List[pipeMTAsyncHandle] = []
        flatten_args_kwargs, _ = tree_flatten((args, kwargs))
        for arg in flatten_args_kwargs:
            if isinstance(arg, pipeMTAsyncHandle):
                assert not arg.result_used, "Using async handle multiple times is not allowed"
                arg.result_used = True
                self.input_handles.append(arg)
            elif isinstance(arg, torch.Tensor) and not arg.is_pinned():
                warnings.warn('''
[pipeMT WARNING] Pageable tensor detected in model input, this could cause performance degradation.
[pipeMT WARNING] Please set pin_memory = True when creating input tensor or data loader.''')
        
        self.input_args, self.input_kwargs = split_args_kwargs_into_chunks(
                                                args, kwargs,
                                                num_microbatch,
                                                args_chunk_spec, kwargs_chunk_spec)
        self.num_microbatch = len(self.input_args)
        for handle in self.input_handles:
            assert handle.input.num_microbatch == self.num_microbatch, \
                'Number of microbatch from async handle should be the same as other input arguments'
        
        self.result_chunk_spec = result_chunk_spec
    
    def peek(self) -> Tuple[List, Dict[str, Any]]:
        flatten_input_cpu, flatten_spec = tree_flatten((self.input_args[0], self.input_kwargs[0]))
        for i in range(len(flatten_input_cpu)):
            if isinstance(flatten_input_cpu[i], pipeMTAsyncHandle):
                flatten_input_cpu[i] = tree_unflatten(flatten_input_cpu[i].flatten_states[0], flatten_input_cpu[i].flatten_specs[0])
        return tree_unflatten(flatten_input_cpu, flatten_spec)
    
    def is_data_ready(self) -> bool:
        for handle in self.input_handles:
            if not handle.is_ready():
                return False
        return True

    def flatten(self) -> Tuple[List[Tuple[Any, ...]], List[TreeSpec], List[Tuple[List[torch.cuda.Event], ...]]]:
        flatten_states = []
        flatten_specs = []
        transfer_events = []
        
        for batch_idx, arg_kwarg in enumerate(zip(self.input_args, self.input_kwargs)):
            forward_event = []
            backward_event = []
            flatten_input, flatten_spec = tree_flatten(arg_kwarg)
            for item in flatten_input:
                if isinstance(item, torch.Tensor):
                    backward_event.append(None)
                    break
            for idx, item in enumerate(flatten_input):
                if isinstance(item, pipeMTAsyncHandle):
                    flatten_input[idx] = tree_unflatten(item.flatten_states[batch_idx], item.flatten_specs[batch_idx])
                    forward_event += item.transfer_events[batch_idx][0]
                    backward_event += item.transfer_events[batch_idx][1]
            flatten_input, flatten_spec = tree_flatten(tree_unflatten(flatten_input, flatten_spec))
            flatten_states.append(flatten_input)
            transfer_events.append((forward_event, backward_event))
            flatten_specs.append(flatten_spec)
        
        return flatten_states, transfer_events, flatten_specs

    def gather_result(self, result: List[Any]):
        return merge_chunks(result, self.result_chunk_spec)
