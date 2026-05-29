from __future__ import annotations

from data_sources.mock_gold import get_mock_gold_price_history, get_mock_gold_snapshot
from decision_engine import decide_gold_action
from indicators.technical import calculate_technical_indicators
from scoring.gold import calculate_gold_score


def test_gold_score_is_deterministic_for_mock_data() -> None:
    snapshot = get_mock_gold_snapshot()
    technical = calculate_technical_indicators(get_mock_gold_price_history())
    assert technical is not None

    score = calculate_gold_score(snapshot, technical)

    assert score.macro_score == 15
    assert score.flow_score == 20
    assert score.total_score == 8
    assert score.signal == "neutral"
    assert "避险情绪较高，提升黄金配置价值" in score.factors


def test_decision_engine_maps_scores_to_expected_action() -> None:
    snapshot = get_mock_gold_snapshot()
    technical = calculate_technical_indicators(get_mock_gold_price_history())
    assert technical is not None
    score = calculate_gold_score(snapshot, technical)

    assert decide_gold_action(score) == "观望"
    assert decide_gold_action(None) == "数据不足"
