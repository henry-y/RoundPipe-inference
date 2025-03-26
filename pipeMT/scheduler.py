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

def model_enqueue(handle: 'pipeMTAsyncHandle'):
    with model_queue_lock:
        model_queue.add(handle)
    scheduler_wake_up.set()

def scheduler_thread():
    while True:
        device = device_queue.get()
        while True:
            handle_to_exec = None
            with model_queue_lock:
                for handle in model_queue:
                    if handle.input.is_data_ready() and handle.is_prior_to(handle_to_exec):
                        handle_to_exec = handle
                if handle_to_exec is not None:
                    model_queue.remove(handle_to_exec)
                    break
            scheduler_wake_up.wait()
            scheduler_wake_up.clear()
        
        device.launch_layer(handle_to_exec)

Thread(target = scheduler_thread, daemon = True,
       name = 'pipeMT Scheduler Thread').start()
