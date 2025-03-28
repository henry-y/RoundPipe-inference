from typing import *
import threading

import torch

from pipeMT.scheduler import device_queue, model_enqueue, scheduler_wake_up
from pipeMT.activation import download_hidden_state, upload_hidden_state
from pipeMT.profile import annotate
from pipeMT.parameter import upload_layer, free_layer_gpu

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
        
        self.order_tag = torch.empty(0, device = device)
        self.detach_tag = threading.Event()
        self.compute_start = torch.cuda.Event()
        
        threading.Thread(target = self.controller_thread, daemon = True,
                         name = f'pipeMT {device} Device Controller Thread').start()
    
    def controller_thread(self):
        while True:
            self.is_active.wait()
            handle = self.active_layer
            
            if self.detach_tag.is_set():
                self.order_tag = self.order_tag.detach()
                self.detach_tag.clear()
            if handle.cur_layer == 0:
                handle.flatten_input()
            
            upload_layer(handle.model.layers[handle.cur_layer], self.upstream, self.compute_stream)
            
            for i in range(handle.input.num_microbatch):
                with annotate(f'{handle.model.name}L{handle.cur_layer}B{i}'):
                    hidden_state = upload_hidden_state(self, handle.transfer_events[i],
                                        handle.flatten_states[i], handle.flatten_specs[i])
                    
                    if i == 0:
                        self.compute_start.record(self.compute_stream)
                    with torch.enable_grad() if handle.require_grad else torch.no_grad():
                        with torch.cuda.stream(self.compute_stream):
                            if handle.cur_layer == 0:
                                args, kwargs = hidden_state
                                hidden_state = handle.model.layers[handle.cur_layer].forward(
                                                    *args, **kwargs)
                            else:
                                hidden_state = handle.model.layers[handle.cur_layer].forward(hidden_state)
                        
                    handle.transfer_events[i], handle.flatten_states[i], handle.flatten_specs[i] = download_hidden_state(self, hidden_state)
            
            free_layer_gpu(handle.model.layers[handle.cur_layer], self.downstream)
            
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
