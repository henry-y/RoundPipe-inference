from typing import *
import threading

import torch

from pipeMT.scheduler import device_queue, model_enqueue, scheduler_wake_up
from pipeMT.profile import annotate
from pipeMT.run import CheckpointRun

if TYPE_CHECKING:
    from pipeMT.async_handle import pipeMTAsyncHandle

class DeviceManager:
    def __init__(self, device: torch.device):
        self.device = device
        self.is_active = threading.Event()
        self.is_idle = threading.Event()
        self.is_idle.set()
        self.active_layer: 'pipeMTAsyncHandle' = None
        
        self.upstream = torch.cuda.Stream(device)
        self.compute_stream = torch.cuda.Stream(device)
        self.downstream = torch.cuda.Stream(device)
        
        self.order_tag = torch.empty(0, requires_grad = True)
        self.detach_tag = threading.Event()
        self.compute_start = torch.cuda.Event()
        
        threading.Thread(target = self.controller_thread, daemon = True,
                         name = f'pipeMT {device} Device Controller Thread').start()
    
    def controller_thread(self):
        while True:
            self.is_active.wait()
            handle = self.active_layer
            
            if self.detach_tag.is_set():
                self.order_tag = torch.empty(0, requires_grad = True)
                self.detach_tag.clear()
            if handle.cur_layer == 0:
                handle.flatten_input()
            
            layer_require_grad = any(handle.model.layer_has_param[handle.cur_layer + i] for i in range(1))
            
            for i in range(handle.input.num_microbatch):
                input_requrie_grad = any(isinstance(t, torch.Tensor) and t.requires_grad for t in handle.flatten_states[i])
                with annotate(f'{handle.model.name}L{handle.cur_layer}B{i}'):
                    with torch.enable_grad() if handle.require_grad else torch.no_grad():
                        order_tag = self.order_tag if layer_require_grad or input_requrie_grad else torch.empty(0)
                        order_tag, *handle.flatten_states[i] \
                            = CheckpointRun.apply(self, handle, 1, i, order_tag, *handle.flatten_states[i])
                        if order_tag.requires_grad:
                            self.order_tag = order_tag
            
            handle.parameter_processed += handle.model.layer_size[handle.cur_layer]
            handle.cur_layer += 1
            with handle.lock:
                handle.prefetch_layer += 1
            if handle.cur_layer < handle.model.num_layers:
                model_enqueue(handle)
            else:
                handle.all_launched.set()
                scheduler_wake_up.set()
            self.is_active.clear()
            self.is_idle.set()
            
            self.compute_start.synchronize()
            with handle.lock:
                handle.prefetch_layer -= 1
            
            self.upstream.synchronize()
            self.active_layer = handle = None
            device_queue.put(self)
    
    def launch_layer(self, handle: 'pipeMTAsyncHandle'):
        self.active_layer = handle
        self.is_idle.clear()
        self.is_active.set()

device_list: List[DeviceManager] = []

for i in range(torch.cuda.device_count()):
    device = DeviceManager(torch.device(f"cuda:{i}"))
    device_list.append(device)
    device_queue.put(device)

def device_tag_detach():
    for device in device_list:
        device.detach_tag.set()
