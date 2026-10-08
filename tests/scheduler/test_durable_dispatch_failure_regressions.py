"""Regressions for one-shot ownership and durable retry backoff."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from apscheduler.events import JobEvent
from apscheduler.triggers.date import DateTrigger

from infrastructure.scheduling.scheduler import apscheduler_executor as scheduler_executor
from infrastructure.scheduling.scheduler.storage import database
from infrastructure.scheduling.scheduler.storage.recoverable_keys import (
    get_recoverable_runs_for_scope,
)
from infrastructure.scheduling.scheduler.storage.run_store import RecoverableRun


class _Scheduler:
    def __init__(self, jobs: list[SimpleNamespace] | None = None) -> None:
        self.jobs = {job.id: job for job in jobs or []}
        self.event_codes: list[int] = []

    def _create_lock(self) -> threading.RLock:
        return threading.RLock()

    def _dispatch_event(self, event: JobEvent) -> None:
        self.event_codes.append(event.code)

    def get_jobs(self) -> list[SimpleNamespace]:
        return list(self.jobs.values())

    def get_job(self, job_id: str) -> SimpleNamespace | None:
        return self.jobs.get(job_id)


def _job(
    task_id: str,
    runners: object,
    *,
    trigger: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=task_id,
        max_instances=1,
        misfire_grace_time=None,
        func=lambda **_kwargs: None,
        args=(task_id, runners),
        kwargs={},
        _jobstore_alias="default",
        trigger=trigger or SimpleNamespace(),
    )


def _silence_observability(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scheduler_executor,
        "_backlog_snapshot",
        lambda _eligible: SimpleNamespace(
            pending_count=0,
            oldest_pending_age_seconds=None,
        ),
    )
    monkeypatch.setattr(
        scheduler_executor,
        "_record_backlog_state",
        lambda *_args, **_kwargs: None,
    )


def test_one_shot_hint_is_exact_and_retires_after_terminal_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed DateTrigger admission cannot authorize a later fire time."""
    runners = object()
    fire_at = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
    first_key = scheduler_executor._durable_fire_time(fire_at)
    later_key = scheduler_executor._durable_fire_time(fire_at + timedelta(minutes=5))
    job = _job(
        "task-once",
        runners,
        trigger=DateTrigger(run_date=fire_at),
    )
    scheduler = _Scheduler()
    pending: list[SimpleNamespace] = []
    executed: list[str] = []
    terminal_checked = threading.Event()
    later_executed = threading.Event()

    def on_submit(task_id: str, scheduled_run_time: datetime) -> None:
        pending.append(
            SimpleNamespace(
                task_id=task_id,
                fire_time=scheduler_executor._durable_fire_time(scheduled_run_time),
            )
        )

    def recoverable_scope_runs(
        eligible_task_ids: set[str],
        exact_run_keys: set[tuple[str, str]],
        *,
        limit: int,
    ) -> list[SimpleNamespace]:
        return [
            run
            for run in pending
            if run.task_id in eligible_task_ids or (run.task_id, run.fire_time) in exact_run_keys
        ][:limit]

    def execute_recoverable(run: SimpleNamespace, _runners: object) -> list[object]:
        pending.remove(run)
        executed.append(run.fire_time)
        if run.fire_time == later_key:
            later_executed.set()
        return []

    def run_is_terminal(task_id: str, fire_time: str) -> bool:
        assert task_id == job.id
        if fire_time == first_key:
            terminal_checked.set()
            return True
        return False

    monkeypatch.setattr(scheduler_executor, "_enabled_task_ids", lambda: {job.id})
    monkeypatch.setattr(scheduler_executor, "_recoverable_scope_runs", recoverable_scope_runs)
    monkeypatch.setattr(scheduler_executor, "_execute_recoverable_run", execute_recoverable)
    monkeypatch.setattr(scheduler_executor, "_durable_run_is_terminal", run_is_terminal)
    _silence_observability(monkeypatch)

    executor = scheduler_executor.ScheduledThreadPoolExecutor(
        max_workers=1,
        on_submit=on_submit,
    )
    executor.start(scheduler, "default")
    recovery_job = SimpleNamespace(id="scheduler-claim-recovery", max_instances=1)
    try:
        executor.submit_job(job, [fire_at])
        assert terminal_checked.wait(5)
        assert executed == [first_key]

        pending.append(SimpleNamespace(task_id=job.id, fire_time=later_key))
        executor.submit_job(recovery_job, [fire_at + timedelta(minutes=6)])
        assert not later_executed.wait(0.5)
        assert executed == [first_key]
        assert [run.fire_time for run in pending] == [later_key]
    finally:
        executor.shutdown(wait=True)


def test_exact_one_shot_scope_filters_before_limit(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Older rows for the same task cannot hide an exact one-shot key."""
    db_path = tmp_path / "runs.db"
    monkeypatch.setattr(database, "default_run_database_path", lambda: db_path)
    exact_fire_time = "2026-09-17T13:00:00Z"

    rows = [
        (
            "task-once",
            f"older-{index:04d}",
            f"2026-09-17T12:{index // 60:02d}:{index % 60:02d}+00:00",
            "pending",
        )
        for index in range(1_000)
    ]
    rows.append(
        (
            "task-once",
            exact_fire_time,
            "2026-09-18T00:00:00+00:00",
            "pending",
        )
    )
    with database.transaction(db_path) as conn:
        conn.executemany(
            "INSERT INTO task_runs (task_id, fire_time, started_at, status) VALUES (?, ?, ?, ?)",
            rows,
        )

    recovered = get_recoverable_runs_for_scope(
        set(),
        {("task-once", exact_fire_time)},
        limit=1,
    )

    assert recovered == [RecoverableRun("task-once", exact_fire_time)]


def test_durable_exception_defers_task_until_periodic_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-claim failure cannot hot-loop the same pending task."""
    runners = object()
    job = _job("task-retry", runners)
    scheduler = _Scheduler([job])
    run = SimpleNamespace(
        task_id=job.id,
        fire_time="2026-09-17T12:00:00Z",
    )
    pending = [run]
    attempts: list[str] = []
    first_attempt = threading.Event()
    second_attempt = threading.Event()

    def recoverable_runs(
        eligible_task_ids: set[str],
        *,
        limit: int,
    ) -> list[SimpleNamespace]:
        return [item for item in pending if item.task_id in eligible_task_ids][:limit]

    def execute_recoverable(item: SimpleNamespace, _runners: object) -> list[object]:
        attempts.append(item.fire_time)
        if len(attempts) == 1:
            first_attempt.set()
            raise RuntimeError("transient storage failure")
        pending.remove(item)
        second_attempt.set()
        return []

    monkeypatch.setattr(scheduler_executor, "_enabled_task_ids", lambda: {job.id})
    monkeypatch.setattr(scheduler_executor, "_recoverable_runs", recoverable_runs)
    monkeypatch.setattr(scheduler_executor, "_execute_recoverable_run", execute_recoverable)
    _silence_observability(monkeypatch)

    executor = scheduler_executor.ScheduledThreadPoolExecutor(
        max_workers=1,
        on_submit=lambda *_args: None,
    )
    executor.start(scheduler, "default")
    recovery_job = SimpleNamespace(id="scheduler-claim-recovery", max_instances=1)
    try:
        executor.submit_job(recovery_job, [datetime.now(UTC)])
        assert first_attempt.wait(5)

        for _ in range(100):
            with executor._dispatch_lock:
                deferred = job.id in executor._deferred_task_ids
            if deferred:
                break
            threading.Event().wait(0.01)
        assert deferred
        threading.Event().wait(0.1)
        assert attempts == [run.fire_time]
        assert pending == [run]

        executor.submit_job(recovery_job, [datetime.now(UTC) + timedelta(minutes=1)])
        assert second_attempt.wait(5)
        assert attempts == [run.fire_time, run.fire_time]
        assert not pending
    finally:
        executor.shutdown(wait=True)


def test_failed_task_does_not_block_unrelated_pending_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed durable row is deferred while the released slot drains other work."""
    runners = object()
    failed_job = _job("task-failed", runners)
    healthy_job = _job("task-healthy", runners)
    scheduler = _Scheduler([failed_job, healthy_job])
    failed_run = SimpleNamespace(
        task_id=failed_job.id,
        fire_time="2026-09-17T12:00:00Z",
    )
    healthy_run = SimpleNamespace(
        task_id=healthy_job.id,
        fire_time="2026-09-17T12:01:00Z",
    )
    pending = [failed_run, healthy_run]
    attempts: list[str] = []
    failed_attempted = threading.Event()
    healthy_completed = threading.Event()

    def recoverable_runs(
        eligible_task_ids: set[str],
        *,
        limit: int,
    ) -> list[SimpleNamespace]:
        return [item for item in pending if item.task_id in eligible_task_ids][:limit]

    def execute_recoverable(item: SimpleNamespace, _runners: object) -> list[object]:
        attempts.append(item.task_id)
        if item.task_id == failed_job.id:
            failed_attempted.set()
            raise RuntimeError("transient storage failure")
        pending.remove(item)
        healthy_completed.set()
        return []

    monkeypatch.setattr(
        scheduler_executor,
        "_enabled_task_ids",
        lambda: {failed_job.id, healthy_job.id},
    )
    monkeypatch.setattr(scheduler_executor, "_recoverable_runs", recoverable_runs)
    monkeypatch.setattr(scheduler_executor, "_execute_recoverable_run", execute_recoverable)
    _silence_observability(monkeypatch)

    executor = scheduler_executor.ScheduledThreadPoolExecutor(
        max_workers=1,
        on_submit=lambda *_args: None,
    )
    executor.start(scheduler, "default")
    recovery_job = SimpleNamespace(id="scheduler-claim-recovery", max_instances=1)
    try:
        executor.submit_job(recovery_job, [datetime.now(UTC)])
        assert failed_attempted.wait(5)
        assert healthy_completed.wait(5)

        with executor._dispatch_lock:
            assert failed_job.id in executor._deferred_task_ids
        assert attempts == [failed_job.id, healthy_job.id]
        assert pending == [failed_run]
    finally:
        executor.shutdown(wait=True)


def test_periodic_recovery_finishes_cancelled_sqlite_ticks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The control wake retains main's deleted-task cleanup behavior."""
    import time

    from infrastructure.scheduling.scheduler import runner
    from infrastructure.scheduling.scheduler.storage import (
        get_latest_run_for_fire_time,
        try_queue_run,
    )
    from infrastructure.scheduling.scheduler.types import TaskStatus

    db_path = tmp_path / "cancelled.db"
    monkeypatch.setattr(database, "default_run_database_path", lambda: db_path)
    monkeypatch.setattr(runner, "list_tasks", lambda: [])
    monkeypatch.setattr(runner, "get_task", lambda _task_id: None)
    monkeypatch.setattr(
        runner, "record_scheduler_service_operation", lambda *_args, **_kwargs: None
    )
    _silence_observability(monkeypatch)
    fire_time = "2026-09-17T12:00:00Z"
    assert try_queue_run("deleted-task", fire_time)

    scheduler = _Scheduler()
    executor = scheduler_executor.ScheduledThreadPoolExecutor(
        max_workers=1, on_submit=lambda _task_id, _scheduled: None
    )
    executor.start(scheduler, "default")
    recovery = _job("scheduler-claim-recovery", object())
    try:
        executor.submit_job(recovery, [datetime.now(UTC)])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            run = get_latest_run_for_fire_time("deleted-task", fire_time)
            if run is not None and run.status is TaskStatus.SKIPPED:
                break
            time.sleep(0.01)
        else:
            pytest.fail("Deleted pending tick was not terminalized")
    finally:
        executor.shutdown(wait=True)


def test_durable_dispatch_emits_apscheduler_execution_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Durable execution still notifies APScheduler execution listeners."""
    import time

    from apscheduler.events import EVENT_JOB_EXECUTED

    fire_at = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
    pending = [RecoverableRun("task-a", fire_at.isoformat())]
    scheduler = _Scheduler([_job("task-a", object())])

    def select_runs(eligible_task_ids: set[str], *, limit: int) -> list[RecoverableRun]:
        return [r for r in pending if r.task_id in eligible_task_ids][:limit]

    def run(_run: RecoverableRun, _runners: object) -> list[object]:
        pending.clear()
        return []

    monkeypatch.setattr(scheduler_executor, "_enabled_task_ids", lambda: {"task-a"})
    monkeypatch.setattr(scheduler_executor, "_recoverable_runs", select_runs)
    monkeypatch.setattr(scheduler_executor, "_execute_recoverable_run", run)
    _silence_observability(monkeypatch)
    executor = scheduler_executor.ScheduledThreadPoolExecutor(
        max_workers=1, on_submit=lambda _task_id, _scheduled: None
    )
    executor.start(scheduler, "default")
    try:
        executor.submit_job(scheduler.jobs["task-a"], [fire_at])
        deadline = time.monotonic() + 5
        while EVENT_JOB_EXECUTED not in scheduler.event_codes and time.monotonic() < deadline:
            time.sleep(0.01)
        assert EVENT_JOB_EXECUTED in scheduler.event_codes
    finally:
        executor.shutdown(wait=True)


def test_durable_dispatch_emits_apscheduler_error_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Execution failures notify listeners, not only executor logging."""
    import time

    from apscheduler.events import EVENT_JOB_ERROR

    fire_at = datetime(2026, 9, 17, 12, 0, tzinfo=UTC)
    pending = [RecoverableRun("task-a", fire_at.isoformat())]
    scheduler = _Scheduler([_job("task-a", object())])

    def select_runs(eligible_task_ids: set[str], *, limit: int) -> list[RecoverableRun]:
        return [r for r in pending if r.task_id in eligible_task_ids][:limit]

    def fail(_run: RecoverableRun, _runners: object) -> list[object]:
        pending.clear()
        raise RuntimeError("simulated durable dispatch failure")

    monkeypatch.setattr(scheduler_executor, "_enabled_task_ids", lambda: {"task-a"})
    monkeypatch.setattr(scheduler_executor, "_recoverable_runs", select_runs)
    monkeypatch.setattr(scheduler_executor, "_execute_recoverable_run", fail)
    _silence_observability(monkeypatch)
    executor = scheduler_executor.ScheduledThreadPoolExecutor(
        max_workers=1, on_submit=lambda _task_id, _scheduled: None
    )
    executor.start(scheduler, "default")
    try:
        executor.submit_job(scheduler.jobs["task-a"], [fire_at])
        deadline = time.monotonic() + 5
        while EVENT_JOB_ERROR not in scheduler.event_codes and time.monotonic() < deadline:
            time.sleep(0.01)
        assert EVENT_JOB_ERROR in scheduler.event_codes
    finally:
        executor.shutdown(wait=True)
