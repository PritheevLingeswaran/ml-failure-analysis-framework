from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageTiming:
    stage_name: str
    start_ts: str
    end_ts: str
    duration_sec: float


class TimingCollector:
    """Collect stage timings as part of the evaluation audit trail.

    Why:
    - Audit artifacts need raw start/end timestamps, not just aggregates.
    - Aggregated totals are still useful for logging and API payloads.
    """

    def __init__(self) -> None:
        self._events: List[StageTiming] = []

    def stage(self, stage_name: str) -> "StageTimer":
        return StageTimer(stage_name=stage_name, collector=self)

    def add(self, timing: StageTiming) -> None:
        self._events.append(timing)

    @property
    def events(self) -> List[StageTiming]:
        return list(self._events)

    def events_payload(self) -> List[Dict[str, Any]]:
        return [asdict(event) for event in self._events]

    def stage_totals(self) -> Dict[str, float]:
        totals: Dict[str, float] = {}
        for event in self._events:
            totals[event.stage_name] = round(totals.get(event.stage_name, 0.0) + event.duration_sec, 6)
        return totals


class StageTimer:
    def __init__(self, stage_name: str, collector: TimingCollector):
        self.stage_name = stage_name
        self.collector = collector
        self._start_ts: str | None = None
        self._start_perf: float | None = None

    def __enter__(self) -> "StageTimer":
        self._start_ts = _utc_now_iso()
        self._start_perf = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start_ts is None or self._start_perf is None:
            return
        end_ts = _utc_now_iso()
        duration_sec = round(perf_counter() - self._start_perf, 6)
        self.collector.add(
            StageTiming(
                stage_name=self.stage_name,
                start_ts=self._start_ts,
                end_ts=end_ts,
                duration_sec=duration_sec,
            )
        )
