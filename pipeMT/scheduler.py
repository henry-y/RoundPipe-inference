from typing import *
from queue import Queue, Empty
from threading import Thread, Event, Lock

if TYPE_CHECKING:
    from pipeMT.async_handle import pipeMTAsyncHandle
    from pipeMT.device import DeviceManager

model_queue: Set['pipeMTAsyncHandle'] = set()
model_queue_lock = Lock()
device_queue: Queue['DeviceManager'] = Queue()
scheduler_wake_up = Event()
scheduling_size = 0

def is_prior_to(self: 'pipeMTAsyncHandle', other: Optional['pipeMTAsyncHandle']) -> bool:
    if other is None:
        return True
    if self.prefetch_layer != other.prefetch_layer:
        return self.prefetch_layer < other.prefetch_layer
    return self.workload_to_proccess - self.workload_processed \
            > other.workload_to_proccess - other.workload_processed

def model_enqueue(handle: 'pipeMTAsyncHandle'):
    global scheduling_size
    with model_queue_lock:
        model_queue.add(handle)
        scheduling_size = max(scheduling_size, handle.model.max_layer_workload)
    scheduler_wake_up.set()

def scheduler_thread():
    from pipeMT.device import device_list
    while True:
        device = device_queue.get()
        while True:
            handle_to_exec = None
            with model_queue_lock:
                for handle in model_queue:
                    if handle.input.is_data_ready() and is_prior_to(handle, handle_to_exec):
                        handle_to_exec = handle
                if handle_to_exec is not None:
                    model_queue.remove(handle_to_exec)
                    break
            scheduler_wake_up.wait()
            scheduler_wake_up.clear()
        
        if handle_to_exec.model.require_spliting:
            args, kwargs = handle_to_exec.input.peek()
            for d in device_list:
                d.is_idle.wait()
            handle_to_exec.model.split_model(args, kwargs)
        device.launch_layer(handle_to_exec)

Thread(target = scheduler_thread, daemon = True,
       name = 'pipeMT Scheduler Thread').start()
