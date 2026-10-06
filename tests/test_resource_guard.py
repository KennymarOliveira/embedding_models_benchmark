import pytest

from app.core.resource_guard import ResourceLimitExceeded, ResourceSnapshot, SystemResourceGuard
from app.services.benchmark_service import BenchmarkExecutionLogger


@pytest.fixture
def guard():
    instance = SystemResourceGuard()
    yield instance
    instance.stop()


def snapshot(ram=50, disk=20, space=50, commit=50, paging=None, process_ram=50):
    return ResourceSnapshot(ram, 4096, 0, space, disk, commit, paging, 4096, process_ram)


@pytest.mark.parametrize("reading", [
    snapshot(ram=100), snapshot(disk=100), snapshot(space=100),
    snapshot(ram=100, disk=None, commit=None),
])
def test_single_resource_pressure_does_not_interrupt(guard, monkeypatch, reading):
    monkeypatch.setattr(guard, "_snapshot", lambda: reading)
    for _ in range(6):
        guard.sample_once()
        guard.checkpoint()
    assert guard.metrics()["ram_warning_emitted"] == (reading.system_ram_percent >= 95)


def test_system_disk_pressure_does_not_interrupt_without_model_memory(guard, monkeypatch):
    monkeypatch.setattr(guard, "_snapshot", lambda: snapshot(ram=99, disk=100, process_ram=10))
    monkeypatch.setattr("app.core.resource_guard.time.monotonic", lambda: 100)
    guard._monitoring_started_at = 0
    for _ in range(6):
        guard.sample_once()
        guard.checkpoint()


def test_model_pressure_must_be_consecutive_after_load_grace(guard, monkeypatch):
    readings = iter([snapshot(ram=95, disk=95)] * 4 + [snapshot(ram=95, disk=10)]
                    + [snapshot(ram=95, disk=95)] * 5)
    monkeypatch.setattr(guard, "_snapshot", lambda: next(readings))
    monkeypatch.setattr("app.core.resource_guard.time.monotonic", lambda: 100)
    guard._monitoring_started_at = 0
    for _ in range(9):
        guard.sample_once()
        guard.checkpoint()
    guard.sample_once()
    with pytest.raises(ResourceLimitExceeded, match="Processo do modelo.*atividade do disco"):
        guard.checkpoint()


def test_commit_limit_interrupts_without_disk_pressure(guard, monkeypatch):
    monkeypatch.setattr(guard, "_snapshot", lambda: snapshot(commit=94.9))
    guard.sample_once()
    guard.checkpoint()
    monkeypatch.setattr(guard, "_snapshot", lambda: snapshot(commit=95))
    guard.sample_once()
    with pytest.raises(ResourceLimitExceeded, match="Memória comprometida"):
        guard.checkpoint()
    assert guard.metrics()["commit_peak_percent"] == 95
    monkeypatch.setattr(guard, "_snapshot", lambda: snapshot())
    guard.sample_once()
    with pytest.raises(ResourceLimitExceeded):
        guard.checkpoint()


@pytest.mark.parametrize("paging, expected", [(None, None), (0, False), (350, True)])
def test_paging_availability_and_activity(guard, monkeypatch, paging, expected):
    monkeypatch.setattr(guard, "_snapshot", lambda: snapshot(paging=paging))
    guard.sample_once()
    assert guard.metrics()["paging_activity_detected"] is expected
    assert guard.metrics()["paging_peak_pages_per_second"] == paging


def test_ram_warning_is_not_logged_every_sample(guard, monkeypatch, caplog):
    readings = iter([snapshot(ram=96)] * 3 + [snapshot(ram=90), snapshot(ram=96)])
    monkeypatch.setattr(guard, "_snapshot", lambda: next(readings))
    for _ in range(5):
        guard.sample_once()
    assert len([record for record in caplog.records if "RAM do sistema" in record.message]) == 2


def test_sample_callback_receives_each_snapshot(monkeypatch):
    received = []
    instance = SystemResourceGuard(sample_callback=received.append)
    reading = snapshot(ram=72, disk=18, process_ram=12)
    monkeypatch.setattr(instance, "_snapshot", lambda: reading)
    try:
        assert instance.sample_once() == reading
        assert received == [reading]
    finally:
        instance.stop()


def test_execution_logger_persists_event_and_resource_sample(tmp_path):
    execution_logger = BenchmarkExecutionLogger(tmp_path, "test-run")
    reading = snapshot(ram=72, disk=18, process_ram=12)

    execution_logger.event("model_started", "mock-model", pid=123)
    execution_logger.resource_sample("mock-model", reading)

    assert '"event": "model_started"' in (tmp_path / "execution.log").read_text(encoding="utf-8")
    sample = (tmp_path / "resource_samples.jsonl").read_text(encoding="utf-8")
    assert '"benchmark_id": "test-run"' in sample
    assert '"model": "mock-model"' in sample
