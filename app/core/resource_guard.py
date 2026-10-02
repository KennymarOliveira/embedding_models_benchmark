"""Monitoramento de recursos do sistema durante a execução de um modelo."""

from __future__ import annotations

import threading
import time
import sys
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from typing import Dict, Optional

import psutil

from app.core.config import (
    RESOURCE_GUARD_CONSECUTIVE_SAMPLES,
    RESOURCE_GUARD_DISK_PATH,
    RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS,
    RESOURCE_GUARD_STOP_PERCENT,
)


class ResourceLimitExceeded(RuntimeError):
    """Indica que a execução deve parar para preservar a responsividade do computador."""


@dataclass(frozen=True)
class ResourceSnapshot:
    system_ram_percent: float
    system_ram_used_mb: float
    swap_percent: float
    disk_space_used_percent: float
    disk_activity_percent: Optional[float]


class WindowsDiskActivitySampler:
    """Lê o mesmo contador de atividade de disco exibido pelo Windows."""

    _PDH_FMT_DOUBLE = 0x00000200

    class _CounterValue(ctypes.Structure):
        _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]

    def __init__(self) -> None:
        self._query = ctypes.c_void_p()
        self._counter = ctypes.c_void_p()
        self._pdh = None
        if sys.platform != "win32":
            return
        try:
            self._pdh = ctypes.WinDLL("pdh")
            self._pdh.PdhOpenQueryW.argtypes = [
                ctypes.c_wchar_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self._pdh.PdhAddEnglishCounterW.argtypes = [
                ctypes.c_void_p,
                ctypes.c_wchar_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self._pdh.PdhCollectQueryData.argtypes = [ctypes.c_void_p]
            self._pdh.PdhGetFormattedCounterValue.argtypes = [
                ctypes.c_void_p,
                wintypes.DWORD,
                ctypes.POINTER(wintypes.DWORD),
                ctypes.POINTER(self._CounterValue),
            ]
            self._pdh.PdhCloseQuery.argtypes = [ctypes.c_void_p]
            if self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query)) != 0:
                self.close()
                return
            counter_path = r"\PhysicalDisk(_Total)\% Disk Time"
            if self._pdh.PdhAddEnglishCounterW(
                self._query, counter_path, 0, ctypes.byref(self._counter)
            ) != 0:
                self.close()
                return
            # A primeira coleta inicializa o contador; a seguinte fornece a taxa.
            self._pdh.PdhCollectQueryData(self._query)
        except (AttributeError, OSError):
            self.close()

    def read_percent(self) -> Optional[float]:
        if self._pdh is None or not self._query.value or not self._counter.value:
            return None
        try:
            if self._pdh.PdhCollectQueryData(self._query) != 0:
                return None
            value = self._CounterValue()
            value_type = wintypes.DWORD()
            if self._pdh.PdhGetFormattedCounterValue(
                self._counter,
                self._PDH_FMT_DOUBLE,
                ctypes.byref(value_type),
                ctypes.byref(value),
            ) != 0:
                return None
            return max(0.0, min(100.0, float(value.doubleValue)))
        except OSError:
            return None

    def close(self) -> None:
        if self._pdh is not None and self._query.value:
            self._pdh.PdhCloseQuery(self._query)
        self._query = ctypes.c_void_p()
        self._counter = ctypes.c_void_p()
        self._pdh = None


class SystemResourceGuard:
    """Amostra recursos do sistema e sinaliza limite após leituras consecutivas."""

    def __init__(
        self,
        *,
        sample_interval_seconds: float = RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS,
        stop_percent: float = RESOURCE_GUARD_STOP_PERCENT,
        consecutive_samples: int = RESOURCE_GUARD_CONSECUTIVE_SAMPLES,
        disk_path: str = RESOURCE_GUARD_DISK_PATH,
    ):
        self.sample_interval_seconds = max(0.1, sample_interval_seconds)
        self.stop_percent = stop_percent
        self.consecutive_samples = max(1, consecutive_samples)
        self.disk_path = disk_path
        self._stop_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._reason: Optional[str] = None
        self._consecutive_ram = 0
        self._consecutive_disk = 0
        self._sample_count = 0
        self._ram_peak_percent = 0.0
        self._ram_peak_mb = 0.0
        self._swap_peak_percent = 0.0
        self._disk_space_peak_percent = 0.0
        self._disk_activity_peak_percent: Optional[float] = None
        self._previous_io = None
        self._previous_io_time: Optional[float] = None
        self._windows_disk_activity = WindowsDiskActivitySampler()

    def _snapshot(self) -> ResourceSnapshot:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        try:
            disk_space_percent = float(psutil.disk_usage(self.disk_path).percent)
        except (OSError, ValueError):
            disk_space_percent = 0.0

        # Linux expõe busy_time em psutil. Em sistemas que não o expõem
        # (por exemplo, algumas instalações Windows/macOS), este campo fica
        # indisponível, mas RAM, swap e espaço em disco continuam protegidos.
        disk_activity_percent = self._windows_disk_activity.read_percent()
        io_now = psutil.disk_io_counters()
        now = time.monotonic()
        if (
            disk_activity_percent is None
            and io_now is not None
            and self._previous_io is not None
            and self._previous_io_time is not None
        ):
            busy_now = getattr(io_now, "busy_time", None)
            busy_before = getattr(self._previous_io, "busy_time", None)
            elapsed = now - self._previous_io_time
            if busy_now is not None and busy_before is not None and elapsed > 0:
                disk_activity_percent = max(0.0, min(100.0, (busy_now - busy_before) / (elapsed * 10)))
        self._previous_io = io_now
        self._previous_io_time = now

        return ResourceSnapshot(
            system_ram_percent=float(memory.percent),
            system_ram_used_mb=round(memory.used / (1024**2), 2),
            swap_percent=float(swap.percent),
            disk_space_used_percent=disk_space_percent,
            disk_activity_percent=disk_activity_percent,
        )

    def sample_once(self) -> ResourceSnapshot:
        snapshot = self._snapshot()
        with self._lock:
            self._sample_count += 1
            self._ram_peak_percent = max(self._ram_peak_percent, snapshot.system_ram_percent)
            self._ram_peak_mb = max(self._ram_peak_mb, snapshot.system_ram_used_mb)
            self._swap_peak_percent = max(self._swap_peak_percent, snapshot.swap_percent)
            self._disk_space_peak_percent = max(self._disk_space_peak_percent, snapshot.disk_space_used_percent)
            if snapshot.disk_activity_percent is not None:
                current_peak = self._disk_activity_peak_percent or 0.0
                self._disk_activity_peak_percent = max(current_peak, snapshot.disk_activity_percent)

            if snapshot.system_ram_percent >= self.stop_percent:
                self._consecutive_ram += 1
            else:
                self._consecutive_ram = 0

            disk_percent = max(snapshot.disk_space_used_percent, snapshot.disk_activity_percent or 0.0)
            if disk_percent >= self.stop_percent:
                self._consecutive_disk += 1
            else:
                self._consecutive_disk = 0

            if self._consecutive_ram >= self.consecutive_samples:
                self._reason = (
                    f"Memória do sistema atingiu {snapshot.system_ram_percent:.1f}% por "
                    f"{self.consecutive_samples} medições consecutivas."
                )
                self._stop_event.set()
            elif self._consecutive_disk >= self.consecutive_samples:
                metric = "atividade" if snapshot.disk_activity_percent is not None else "ocupação"
                self._reason = (
                    f"{metric.capitalize()} do disco atingiu {disk_percent:.1f}% por "
                    f"{self.consecutive_samples} medições consecutivas."
                )
                self._stop_event.set()
        return snapshot

    def _monitor(self) -> None:
        while not self._shutdown_event.wait(self.sample_interval_seconds):
            self.sample_once()
            if self._stop_event.is_set():
                return

    def start(self) -> None:
        if self._thread is not None:
            return
        self.sample_once()
        self._thread = threading.Thread(target=self._monitor, name="resource-guard", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._shutdown_event.set()
        if self._thread is not None:
            self._thread.join(timeout=self.sample_interval_seconds + 0.2)
        self._windows_disk_activity.close()

    def checkpoint(self) -> None:
        if self._stop_event.is_set():
            raise ResourceLimitExceeded(self.reason or "Limite de recursos atingido.")

    @property
    def reason(self) -> Optional[str]:
        with self._lock:
            return self._reason

    def metrics(self) -> Dict[str, Optional[float] | int | bool | str | None]:
        with self._lock:
            return {
                "system_ram_peak_percent": round(self._ram_peak_percent, 2),
                "system_ram_peak_mb": round(self._ram_peak_mb, 2),
                "swap_peak_percent": round(self._swap_peak_percent, 2),
                "disk_space_peak_percent": round(self._disk_space_peak_percent, 2),
                "disk_activity_peak_percent": (
                    round(self._disk_activity_peak_percent, 2)
                    if self._disk_activity_peak_percent is not None
                    else None
                ),
                "resource_samples": self._sample_count,
                "interrupted_by_resource_guard": self._stop_event.is_set(),
                "interruption_reason": self._reason,
            }
