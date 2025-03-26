from typing import *

import torch
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks, merge_chunks
from torch.utils._pytree import tree_flatten, tree_unflatten

from pipeMT.async_handle import pipeMTAsyncHandle
from pipeMT.warning import throw

if TYPE_CHECKING:
    from torch.distributed.pipelining.microbatch import TensorChunkSpec

class Batch:
    def __init__(self, *args,
                 num_microbatch: Optional[int] = torch.cuda.device_count() + 1,
                 args_chunk_spec: Optional[Tuple['TensorChunkSpec', ...]] = None,
                 kwargs_chunk_spec: Optional[Dict[str, 'TensorChunkSpec']] = None,
                 result_chunk_spec: Optional[Tuple['TensorChunkSpec', ...]] = None,
                 **kwargs):
        self.input_handles: Set[pipeMTAsyncHandle] = set()
        flatten_args_kwargs, _ = tree_flatten((args, kwargs))
        for arg in flatten_args_kwargs:
            if isinstance(arg, pipeMTAsyncHandle):
                self.input_handles.add(arg)
            elif isinstance(arg, torch.Tensor) and not arg.is_pinned():
                throw('Pageable tensor detected in model input, this could cause performance degradation.')
                throw('Please set pin_memory = True when creating input tensor or data loader.')
        
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
                flatten_input_cpu[i] = flatten_input_cpu[i].flatten_states[0]
        return tree_unflatten(flatten_input_cpu, flatten_spec)
    
    def is_data_ready(self) -> bool:
        for handle in self.input_handles:
            if not handle.is_ready():
                return False
        return True

    def gather_result(self, result: List[Any]):
        return merge_chunks(result, self.result_chunk_spec)
