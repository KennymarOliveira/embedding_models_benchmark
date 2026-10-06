"""Monitoramento de recursos do sistema durante a execução de um modelo."""

from __future__ import annotations

import threading
import time
import sys
import ctypes
import logging
from ctypes import wintypes
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import psutil

from app.core.config import (
    RESOURCE_GUARD_CONSECUTIVE_SAMPLES,
    RESOURCE_GUARD_DISK_PATH,
    RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS,
    RESOURCE_GUARD_STOP_PERCENT,
    RESOURCE_GUARD_WARNING_PERCENT,
    RESOURCE_GUARD_COMMIT_STOP_PERCENT,
    RESOURCE_GUARD_LOAD_GRACE_SECONDS,
    RESOURCE_GUARD_PROCESS_RAM_STOP_PERCENT,
)

logger = logging.getLogger(__name__)


class PerformanceInformation(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [
        (name, ctypes.c_size_t) for name in (
            "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal",
            "PhysicalAvailable", "SystemCache", "KernelTotal", "KernelPaged",
            "KernelNonpaged", "PageSize",
        )
    ] + [(name, wintypes.DWORD) for name in ("HandleCount", "ProcessCount", "ThreadCount")]


def read_commit_percent() -> Optional[float]:
    """Lê compromisso/limite reais do Windows; indisponível não significa zero."""
    if sys.platform != "win32":
        return None
    try:
        info = PerformanceInformation()
        info.cb = ctypes.sizeof(info)
        api = ctypes.WinDLL("psapi").GetPerformanceInfo
        api.argtypes = [ctypes.POINTER(PerformanceInformation), wintypes.DWORD]
        api.restype = wintypes.BOOL
        if api(ctypes.byref(info), info.cb) and info.CommitLimit:
            return 100.0 * info.CommitTotal / info.CommitLimit
    except (AttributeError, OSError):
        pass
    return None


class ResourceLimitExceeded(RuntimeError):
    """Indica que a execução deve parar para preservar a responsividade do computador."""


@dataclass(frozen=True)
class ResourceSnapshot:
    system_ram_percent: float
    system_ram_used_mb: float
    swap_percent: float
    disk_space_used_percent: float
    disk_activity_percent: Optional[float]
    commit_percent: Optional[float] = None
    paging_pages_per_second: Optional[float] = None
    process_ram_mb: Optional[float] = None
    process_ram_percent: Optional[float] = None


class WindowsDiskActivitySampler:
    """Lê o mesmo contador de atividade de disco exibido pelo Windows."""

    _PDH_FMT_DOUBLE = 0x00000200

    class _CounterValue(ctypes.Structure):
        _fields_ = [("CStatus", wintypes.DWORD), ("doubleValue", ctypes.c_double)]

    def __init__(self, counter_path: str = r"\PhysicalDisk(_Total)\% Disk Time", *, cap_percent: bool = True) -> None:
        self._cap_percent = cap_percent
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
            if value.CStatus not in (0, 1):
                return None
            result = max(0.0, float(value.doubleValue))
            return min(100.0, result) if self._cap_percent else result
        except OSError:
            return None

    def close(self) -> None:
        if self._pdh is not None and self._query.value:
            self._pdh.PdhCloseQuery(self._query)
        self._query = ctypes.c_void_p()
        self._counter = ctypes.c_void_p()
        self._pdh = None


class SystemResourceGuard:
    """Monitora o sistema e atribui pressão de RAM/disco ao processo do modelo."""

    def __init__(
        self,
        *,
        sample_interval_seconds: float = RESOURCE_GUARD_SAMPLE_INTERVAL_SECONDS,
        stop_percent: float = RESOURCE_GUARD_STOP_PERCENT,
        warning_percent: float = RESOURCE_GUARD_WARNING_PERCENT,
        commit_stop_percent: float = RESOURCE_GUARD_COMMIT_STOP_PERCENT,
        process_ram_stop_percent: float = RESOURCE_GUARD_PROCESS_RAM_STOP_PERCENT,
        load_grace_seconds: float = RESOURCE_GUARD_LOAD_GRACE_SECONDS,
        consecutive_samples: int = RESOURCE_GUARD_CONSECUTIVE_SAMPLES,
        disk_path: str = RESOURCE_GUARD_DISK_PATH,
        sample_callback: Optional[Callable[[ResourceSnapshot], None]] = None,
    ):
        self.sample_interval_seconds = max(0.1, sample_interval_seconds)
        self.stop_percent = stop_percent
        self.warning_percent = warning_percent
        self.commit_stop_percent = commit_stop_percent
        self.process_ram_stop_percent = process_ram_stop_percent
        self.load_grace_seconds = max(0.0, load_grace_seconds)
        self.consecutive_samples = max(1, consecutive_samples)
        self.disk_path = disk_path
        self._stop_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._reason: Optional[str] = None
        self._ram_warning_active = False
        self._ram_warning_emitted = False
        self._consecutive_disk = 0
        self._sample_count = 0
        self._ram_peak_percent = 0.0
        self._ram_peak_mb = 0.0
        self._swap_peak_percent = 0.0
        self._disk_space_peak_percent = 0.0
        self._disk_activity_peak_percent: Optional[float] = None
        self._commit_peak_percent: Optional[float] = None
        self._paging_peak: Optional[float] = None
        self._process_ram_peak_mb: Optional[float] = None
        self._process_ram_peak_percent: Optional[float] = None
        self._monitored_process: Optional[psutil.Process] = None
        self._monitoring_started_at: Optional[float] = None
        self._sample_callback = sample_callback
        self._previous_io = None
        self._previous_io_time: Optional[float] = None
        self._windows_disk_activity = WindowsDiskActivitySampler()
        self._windows_paging = WindowsDiskActivitySampler(r"\Memory\Pages/sec", cap_percent=False)

    def monitor_process(self, pid: int) -> None:
        """Define o processo filho cuja memória deve justificar a interrupção."""
        try:
            monitored_process = psutil.Process(pid)
        except (psutil.Error, ValueError):
            logger.warning("Não foi possível monitorar o processo filho %s.", pid)
            return
        with self._lock:
            self._monitored_process = monitored_process
            self._monitoring_started_at = time.monotonic()

    def set_sample_callback(self, callback: Optional[Callable[[ResourceSnapshot], None]]) -> None:
        """Define um consumidor opcional das amostras, executado fora do lock interno."""
        with self._lock:
            self._sample_callback = callback

    def _snapshot(self) -> ResourceSnapshot:
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()
        try:
            disk_space_percent = float(psutil.disk_usage(self.disk_path).percent)
        except (OSError, ValueError):
            disk_space_percent = 0.0

        # Contadores indisponíveis ficam como None, sem simular leituras.
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

        process_ram_mb = None
        process_ram_percent = None
        with self._lock:
            monitored_process = self._monitored_process
        if monitored_process is not None:
            try:
                process_ram_mb = round(monitored_process.memory_info().rss / (1024**2), 2)
                process_ram_percent = 100.0 * process_ram_mb * (1024**2) / memory.total
            except (psutil.Error, AttributeError):
                pass

        return ResourceSnapshot(
            system_ram_percent=float(memory.percent),
            system_ram_used_mb=round(memory.used / (1024**2), 2),
            swap_percent=float(swap.percent),
            disk_space_used_percent=disk_space_percent,
            disk_activity_percent=disk_activity_percent,
            commit_percent=read_commit_percent(),
            paging_pages_per_second=self._windows_paging.read_percent(),
            process_ram_mb=process_ram_mb,
            process_ram_percent=process_ram_percent,
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

            if snapshot.commit_percent is not None:
                self._commit_peak_percent = max(self._commit_peak_percent or 0.0, snapshot.commit_percent)
            if snapshot.paging_pages_per_second is not None:
                self._paging_peak = max(self._paging_peak or 0.0, snapshot.paging_pages_per_second)
            if snapshot.process_ram_mb is not None:
                self._process_ram_peak_mb = max(self._process_ram_peak_mb or 0.0, snapshot.process_ram_mb)
            if snapshot.process_ram_percent is not None:
                self._process_ram_peak_percent = max(
                    self._process_ram_peak_percent or 0.0, snapshot.process_ram_percent
                )

            ram_warning = snapshot.system_ram_percent >= self.warning_percent
            if ram_warning and not self._ram_warning_active:
                logger.warning("RAM do sistema em %.1f%%; execução continua sob monitoramento.", snapshot.system_ram_percent)
                self._ram_warning_emitted = True
            self._ram_warning_active = ram_warning

            elapsed = (
                time.monotonic() - self._monitoring_started_at
                if self._monitoring_started_at is not None
                else 0.0
            )
            if (elapsed >= self.load_grace_seconds
                    and snapshot.system_ram_percent >= self.stop_percent
                    and snapshot.disk_activity_percent is not None
                    and snapshot.disk_activity_percent >= self.stop_percent
                    and snapshot.process_ram_percent is not None
                    and snapshot.process_ram_percent >= self.process_ram_stop_percent):
                self._consecutive_disk += 1
            else:
                self._consecutive_disk = 0

            if self._stop_event.is_set():
                callback = self._sample_callback
            elif snapshot.commit_percent is not None and snapshot.commit_percent >= self.commit_stop_percent:
                self._reason = (
                    f"Memória comprometida atingiu {snapshot.commit_percent:.1f}% do limite "
                    "de RAM + paginação."
                )
                self._stop_event.set()
                callback = self._sample_callback
            elif self._consecutive_disk >= self.consecutive_samples:
                self._reason = (
                    f"Processo do modelo em {snapshot.process_ram_percent:.1f}% da RAM; sistema em "
                    f"{snapshot.system_ram_percent:.1f}% e atividade do disco em "
                    f"{snapshot.disk_activity_percent:.1f}% por "
                    f"{self.consecutive_samples} medições consecutivas."
                )
                self._stop_event.set()
                callback = self._sample_callback
            else:
                callback = self._sample_callback
        if callback is not None:
            try:
                callback(snapshot)
            except Exception:
                logger.exception("Falha ao registrar amostra de recursos.")
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
        self._windows_paging.close()

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
                "ram_warning_emitted": self._ram_warning_emitted,
                "commit_peak_percent": round(self._commit_peak_percent, 2) if self._commit_peak_percent is not None else None,
                "paging_peak_pages_per_second": round(self._paging_peak, 2) if self._paging_peak is not None else None,
                "paging_activity_detected": self._paging_peak > 0 if self._paging_peak is not None else None,
                "process_ram_peak_mb": (
                    round(self._process_ram_peak_mb, 2) if self._process_ram_peak_mb is not None else None
                ),
                "process_ram_peak_percent": (
                    round(self._process_ram_peak_percent, 2)
                    if self._process_ram_peak_percent is not None else None
                ),
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
