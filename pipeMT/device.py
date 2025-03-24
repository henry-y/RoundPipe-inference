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
        
        self.parameter_upstream = torch.cuda.Stream(device, 0)
        self.activation_upstream = torch.cuda.Stream(device, -1)
        self.compute_stream = torch.cuda.Stream(device, -2)
        self.activation_downstream = torch.cuda.Stream(device, -1)
        self.parameter_downstream = torch.cuda.Stream(device, 0)
        
        self.order_tag = torch.empty(0, device = device)
        self.detach_tag = threading.Event()
        self.first_batch_start = torch.cuda.Event()
        self.first_batch_finish = torch.cuda.Event()
        
        threading.Thread(target = self.monitor_thread, daemon = True,
                         name = f'pipeMT {device} Device Monitor Thread').start()
    
    def monitor_thread(self):
        while True:
            self.is_active.wait()
            self.first_batch_start.synchronize()
            if self.active_layer.cur_layer < self.active_layer.model.num_layers:
                model_enqueue(self.active_layer)
            else:
                self.active_layer.all_launched.set()
                scheduler_wake_up.set()
            self.first_batch_finish.synchronize()
            self.active_layer = None
            self.is_active.clear()
            device_queue.put(self)
    
    def launch_layer(self, handle: 'pipeMTAsyncHandle'):
        if self.detach_tag.is_set():
            self.order_tag = self.order_tag.detach()
            self.detach_tag.clear()
        if handle.model.require_spliting:
            args, kwargs = handle.input.peek()
            handle.model.split_model(args, kwargs)
        
        upload_layer(handle.model.layers[handle.cur_layer], self.parameter_upstream, self.compute_stream)
        
        for i in range(handle.input.num_microbatch):
            with annotate(f'{handle.model.name}L{handle.cur_layer}B{i}'):
                if handle.cur_layer == 0:
                    args, kwargs = upload_input(self, handle.input, i)
                else:
                    hidden_state = upload_hidden_state(self, handle.transfer_event[i],
                                        handle.flatten_states[i], handle.flatten_specs[i])
                self.activation_upstream.wait_stream(self.compute_stream) # limit prefetch rate
                
                if i == 0:
                    self.first_batch_start.record(self.compute_stream)
                with torch.enable_grad() if handle.require_grad else torch.no_grad():
                    with torch.cuda.stream(self.compute_stream):
                        if handle.cur_layer == 0:
                            hidden_state = handle.model.layers[handle.cur_layer].forward(
                                                *args, **kwargs)
                        else:
                            hidden_state = handle.model.layers[handle.cur_layer].forward(hidden_state)
                if i == 0:
                    self.first_batch_finish.record(self.compute_stream)
                    
                handle.transfer_event[i], handle.flatten_states[i], handle.flatten_specs[i] = download_hidden_state(self, hidden_state)
        
        free_layer_gpu(handle.model.layers[handle.cur_layer], self.parameter_downstream)
        handle.parameter_processed += handle.model.layer_size[handle.cur_layer]
        handle.cur_layer += 1
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
