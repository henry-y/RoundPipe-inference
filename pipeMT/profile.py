import os
import contextlib
from typing import *

PROFILER_TYPE = None

if os.environ.get('NSYS_PROFILING_SESSION_ID'):
    PROFILER_TYPE = 'nsys'
    import nvtx

def annotate(name: str, color: Optional[str] = None):
    if PROFILER_TYPE == 'nsys':
        return nvtx.annotate(name, color = color)
    else:
        return contextlib.nullcontext()
