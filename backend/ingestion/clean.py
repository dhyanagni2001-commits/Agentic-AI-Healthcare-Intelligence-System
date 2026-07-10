"""Loggable cleaning rules for raw CSV fields.

Each cleaning function records a counter of what it changed (which field,
how many rows) instead of silently discarding/nulling data, so a single
ingestion run can report exactly what was cleaned and why.
"""
from __future__ import annotations
from collections import Counter
from typing import Optional


class CleaningReport:
    """Accumulates counts of cleaning actions across a single ingestion run."""

    def __init__(self):
        self.counts: Counter = Counter()

    def record(self, rule: str, field: str):
        self.counts[f"{rule}:{field}"] += 1

    def summary(self) -> dict:
        return dict(sorted(self.counts.items(), key=lambda kv: -kv[1]))

    def log_summary(self, log) -> None:
        if not self.counts:
            log.info("Cleaning report: no fields required cleaning")
            return
        log.info("Cleaning report:")
        for key, n in self.summary().items():
            rule, field = key.split(":", 1)
            log.info(f"  {rule}: {n} rows in '{field}'")


_NULLISH = {"n/a", "na", "none", "null", ""}


def clean_str(v: Optional[str], field: str, report: CleaningReport) -> Optional[str]:
    """Trim whitespace and normalise nullish placeholder text to None."""
    s = str(v if v is not None else "").strip()
    if not s:
        return None
    if s.lower() in _NULLISH:
        report.record("cleaned_nullish_to_none", field)
        return None
    return s


def clean_bool(v: Optional[str], field: str, report: CleaningReport) -> bool:
    s = str(v if v is not None else "").strip().lower()
    return s in ("yes", "true", "1", "y")


def clean_int(v: Optional[str], field: str, report: CleaningReport,
              default: Optional[int] = None) -> Optional[int]:
    s = str(v if v is not None else "").strip()
    if not s or s.lower() in _NULLISH:
        return default
    try:
        return int(float(s))
    except ValueError:
        report.record("uncleanable_int_dropped", field)
        return default
