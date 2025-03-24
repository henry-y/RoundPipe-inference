import os
from typing import *

PROFILER_TYPE = None

if os.environ.get('NSYS_PROFILING_SESSION_ID'):
    PROFILER_TYPE = 'nsys'
    import nvtx

class DummyContext:
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass

def annotate(name: str, color: Optional[str] = None):
    if PROFILER_TYPE == 'nsys':
        return nvtx.annotate(name, color = color)
    else:
        return DummyContext()
