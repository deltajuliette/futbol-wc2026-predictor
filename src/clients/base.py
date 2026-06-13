"""Adapter protocols. Downstream code depends on these, not on concrete sources.

A new source only has to satisfy the relevant Protocol; ETL and models never change.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from clients.types import BenchmarkProbRecord, FixtureRecord, OddsSnapshotRecord


@runtime_checkable
class FixtureSource(Protocol):
    def get_fixtures(self, competition: str, season: str | None = None) -> list[FixtureRecord]:
        ...


@runtime_checkable
class OddsSource(Protocol):
    def get_odds(self, competition: str, season: str | None = None) -> list[OddsSnapshotRecord]:
        ...


@runtime_checkable
class BenchmarkSource(Protocol):
    def get_benchmark_probs(
        self, competition: str, season: str | None = None
    ) -> list[BenchmarkProbRecord]:
        ...
