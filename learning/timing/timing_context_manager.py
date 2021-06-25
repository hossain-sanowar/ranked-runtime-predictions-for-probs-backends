import time
from contextlib import contextmanager

"""
This module is responsible for the simplified documentation of any type of time tracking
"""

current_elapsed_time = None


@contextmanager
def time_context(name, file=None):
    start_time = time.perf_counter()
    yield
    elapsed_time = time.perf_counter() - start_time

    output = '[{}] finished in {}s.'.format(name, elapsed_time)
    print(output, file=None)
    if file is not None:
        print(output, file=file)
        file.flush()
    global current_elapsed_time
    current_elapsed_time = elapsed_time
