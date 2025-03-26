from typing import *
import threading

import torch

from pipeMT.scheduler import device_queue, model_enqueue, scheduler_wake_up
from pipeMT.activation import download_hidden_state, upload_hidden_state, upload_input
from pipeMT.profile import annotate
from pipeMT.parameter import upload_layer, free_layer_gpu

if TYPE_CHECKING:
    from pipeMT.async_handle import pipeMTAsyncHandle

class DeviceManager:
    def __init__(self, device: torch.device):
        self.is_active = threading.Event()
        self.active_layer: 'pipeMTAsyncHandle' = None
        
        self.upstream = torch.cuda.Stream(device)
        self.compute_stream = torch.cuda.Stream(device)
        self.downstream = torch.cuda.Stream(device)
        
        self.order_tag = torch.empty(0, device = device)
        self.detach_tag = threading.Event()
        self.compute_start = torch.cuda.Event()
        
        threading.Thread(target = self.monitor_thread, daemon = True,
                         name = f'pipeMT {device} Device Monitor Thread').start()
    
    def monitor_thread(self):
        while True:
            self.is_active.wait()
            self.compute_start.synchronize()
            with self.active_layer.lock:
                self.active_layer.prefetch_layer -= 1
            self.active_layer = None
            self.upstream.synchronize()
            self.is_active.clear()
            device_queue.put(self)
    
    def launch_layer(self, handle: 'pipeMTAsyncHandle'):
        if self.detach_tag.is_set():
            self.order_tag = self.order_tag.detach()
            self.detach_tag.clear()
        if handle.model.require_spliting:
            args, kwargs = handle.input.peek()
            handle.model.split_model(args, kwargs)
        
        upload_layer(handle.model.layers[handle.cur_layer], self.upstream, self.compute_stream)
        
        for i in range(handle.input.num_microbatch):
            with annotate(f'{handle.model.name}L{handle.cur_layer}B{i}'):
                if handle.cur_layer == 0:
                    args, kwargs = upload_input(self, handle.input, i)
                else:
                    hidden_state = upload_hidden_state(self, handle.transfer_event[i],
                                        handle.flatten_states[i], handle.flatten_specs[i])
                
                if i == 0:
                    self.compute_start.record(self.compute_stream)
                with torch.enable_grad() if handle.require_grad else torch.no_grad():
                    with torch.cuda.stream(self.compute_stream):
                        if handle.cur_layer == 0:
                            hidden_state = handle.model.layers[handle.cur_layer].forward(
                                                *args, **kwargs)
                        else:
                            hidden_state = handle.model.layers[handle.cur_layer].forward(hidden_state)
                    
                handle.transfer_event[i], handle.flatten_states[i], handle.flatten_specs[i] = download_hidden_state(self, hidden_state)
        
        free_layer_gpu(handle.model.layers[handle.cur_layer], self.downstream)
        handle.parameter_processed += handle.model.layer_size[handle.cur_layer]
        handle.cur_layer += 1
        with handle.lock:
            handle.prefetch_layer += 1
        if handle.cur_layer < handle.model.num_layers:
            model_enqueue(handle)
        else:
            handle.all_launched.set()
        self.active_layer = handle
        self.is_active.set()

device_list: List[DeviceManager] = []

for i in range(torch.cuda.device_count()):
    device = DeviceManager(torch.device(f"cuda:{i}"))
    device_list.append(device)
    device_queue.put(device)

def device_tag_detach():
    for device in device_list:
        device.detach_tag.set()
