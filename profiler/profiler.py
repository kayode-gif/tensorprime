import time
import psutil
import os
import traceback
from functools import wraps


class Profiler:
    def __init__(self, profile_memory=False):
        self.profile_memory = profile_memory  # tracks the memory usage of each op
        self.proc = psutil.Process(os.getpid())
        self.results = []

    def profile_op(self, fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            mem_before = self.proc.memory_info().rss if self.profile_memory else None
            cpu_before = self.proc.cpu_times()

            start_time = time.time()
            result = fn(*args, **kwargs)
            end_time = time.time()
            mem_after = self.proc.memory_info().rss if self.profile_memory else None
            cpu_after = self.proc.cpu_times()
            cpu_time_spent = (
                    (cpu_after.user - cpu_before.user) +
                    (cpu_after.system - cpu_before.system)
            )
            shapes = [getattr(arg, 'shape', None) for arg in args if hasattr(arg, 'shape')]
            self.results.append({
                'function': fn.__qualname__,
                'time': end_time - start_time,
                'memory_diff': (mem_after - mem_before) if (mem_before is not None and mem_after is not None) else None,
                'cpu_time': cpu_time_spent,
                'shapes': shapes,
            })
            return result

        return wrapper

    def report(self):
        agg = {}
        for result in self.results:
            func = result['function']
            cpu_time = result['cpu_time']
            wall_time = result['time']
            mem_diff = result['memory_diff']
            shapes = result['shapes']

            if func not in agg:
                agg[func] = {
                    'total_cpu_time': 0.0,
                    'total_wall_time': 0.0,
                    'total_mem': 0.0,
                    'count': 0,
                    'shapes': [],
                }
            agg[func]['total_cpu_time'] += cpu_time
            agg[func]['total_wall_time'] += wall_time
            if mem_diff is not None:
                agg[func]['total_mem'] += mem_diff
            agg[func]['count'] += 1
            if shapes:
                agg[func]['shapes'].append(shapes)

        sorted_funcs = sorted(
            agg.items(),
            key=lambda x: (x[1]['total_cpu_time'] / x[1]['count']),
            reverse=True
        )

        print(
            f"{'Function':<30} {'Calls':<6} {'Avg Wall (ms)':<14} {'Avg CPU (ms)':<14} {'Avg Mem (MB)':<14} {'Shapes':<20}")
        print("-" * 120)
        for func, stats in sorted_funcs:
            count = stats['count']
            avg_wall = (stats['total_wall_time'] / count) * 1000
            avg_cpu = (stats['total_cpu_time'] / count) * 1000
            avg_mem = (stats['total_mem'] / count) / (1024 * 1024) if count else 0
            shape_ops = stats['shapes'][0] if stats['shapes'] else ""
            print(
                f"{func:<30} {count:<6} {avg_wall:<14.3f} {avg_cpu:<14.3f} {avg_mem:<14.3f} {str(shape_ops):<20} ")

