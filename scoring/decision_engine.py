"""Configurable decision mapping for factor-score totals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Decision = Literal["强多", "偏多", "中性", "偏空", "强空"]


@dataclass(frozen=True)
class DecisionThresholds:
    """Configurable total-score thresholds for each decision bucket."""

    strong_bullish: int = 60
    bullish: int = 20
    bearish: int = -20
    strong_bearish: int = -60

    def __post_init__(self) -> None:
        """Validate threshold ordering."""

        if not (
            self.strong_bullish > self.bullish > self.bearish > self.strong_bearish
        ):
            raise ValueError(
                "decision thresholds must satisfy strong_bullish > bullish > "
                "bearish > strong_bearish"
            )


def decide_from_total_score(
    total_score: int, thresholds: DecisionThresholds | None = None
) -> Decision:
    """Map a total score into 强多/偏多/中性/偏空/强空."""

    active_thresholds = thresholds or DecisionThresholds()
    if total_score >= active_thresholds.strong_bullish:
        return "强多"
    if total_score >= active_thresholds.bullish:
        return "偏多"
    if total_score <= active_thresholds.strong_bearish:
        return "强空"
    if total_score <= active_thresholds.bearish:
        return "偏空"
    return "中性"
