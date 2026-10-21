"""GPU utilisation via NVML, degrading to an empty dict where NVML is unavailable."""

from __future__ import annotations

import logging
from typing import Any

import torch

log = logging.getLogger(__name__)


class GpuMonitor:
    def __init__(self, index: int = 0) -> None:
        self._handle: Any = None
        if not torch.cuda.is_available():
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:  # NVML missing or driver mismatch: not fatal
            log.debug("NVML unavailable: %s", exc)

    def sample(self) -> dict[str, float]:
        stats: dict[str, float] = {}
        if torch.cuda.is_available():
            stats["gpu_mem_allocated_gb"] = torch.cuda.memory_allocated() / 2**30
            stats["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
            stats["gpu_mem_reserved_gb"] = torch.cuda.memory_reserved() / 2**30
        if self._handle is None:
            return stats
        try:
            util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
            stats["gpu_util_percent"] = float(util.gpu)
            stats["gpu_temp_c"] = float(
                self._nvml.nvmlDeviceGetTemperature(self._handle, self._nvml.NVML_TEMPERATURE_GPU)
            )
            stats["gpu_power_w"] = self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
        except Exception:  # transient NVML errors are common on laptops
            pass
        return stats
