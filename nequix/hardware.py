import ctypes

import jax


def release_device_memory_pools() -> None:
    """Return each CUDA device mempool's cached-but-free memory to the driver.

    Under ``XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`` XLA frees buffers into
    the device's default mempool but pins its release threshold to the memory
    fraction, so the driver never gets the memory back; trimming hands the
    idle trainer's cache to concurrently launched evaluation workers. It is a
    no-op under the default preallocated BFC pool, which cudart cannot trim.
    """
    lib = ctypes.CDLL("libcudart.so.12")
    for index in range(jax.local_device_count()):
        lib.cudaSetDevice(index)
        pool = ctypes.c_void_p()
        lib.cudaDeviceGetDefaultMemPool(ctypes.byref(pool), index)
        lib.cudaMemPoolTrimTo(pool, ctypes.c_size_t(0))


def peak_device_memory_bytes() -> int:
    """Return the highest peak-memory statistic reported by any JAX device."""
    peaks = []
    for device in jax.devices():
        try:
            memory = device.memory_stats() or {}
        except (AttributeError, RuntimeError):
            continue
        for key in ("peak_bytes_in_use", "peak_bytes_in_use_limit", "bytes_in_use"):
            if key in memory:
                peaks.append(int(memory[key]))
                break
    return max(peaks, default=0)
