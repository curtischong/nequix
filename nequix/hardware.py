import jax


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
